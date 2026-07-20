"""
IndexStore — одна база фотографий: FAISS-индекс + файлы + метаданные.

В отличие от прежнего SearchEngine (bot/inference.py), который умел только открыть готовый
индекс и добавить в него по одной картинке, IndexStore закрывает весь пользовательский
сценарий веб-интерфейса: создать пустую базу, пакетно добавить, удалить, посчитать объём,
отдать файлы на экспорт.

Раскладка папки базы:
    <root>/
        images.index          FAISS IndexFlatIP по эмбеддингам картинок
        images_meta.json      {"version": 3, "photos": [...]} — порядок = порядок векторов
        captions.index        опционально: подписи в пространстве CLIP (есть у COCO-базы)
        captions_meta.json
        captions_sbert.index  опционально: те же подписи в пространстве текстовой модели
        captions_sbert.json   {"version": 1, "model": "...", "rows": [photo_id, ...]}
        translate_cache.json
        photos/               оригиналы, имя файла = <photo_id><ext>
        thumbs/               превью 320px WebP

Инвариант, на котором держится всё остальное: i-я запись в images_meta.json описывает
i-й вектор в images.index. Любая операция, меняющая одно, обязана менять и второе — под
локом и с атомарной записью на диск.

Для captions_sbert.index этот инвариант намеренно НЕ действует. Подписи появляются
позже фотографий и покрывают базу частично: пока генератор не дошёл до снимка,
подписи у него нет. Позиционное соответствие здесь развалилось бы на первом же
удалении снимка без подписи, поэтому строки индекса подписей адресуются не номером,
а собственным списком photo_id в captions_sbert.json.

Индекс подписей вторичен: если он повреждён или отстал от базы, поиск по подписям
выключается, но сама база продолжает открываться и работать.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import faiss
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core.embeddings import ProgressCallback, compute_image_embeddings, open_image
from core.model import ModelHolder
from core.translate import TRANSLATE_CACHE_FILE, maybe_translate

IMAGES_INDEX = "images.index"
IMAGES_META = "images_meta.json"
CAPTIONS_INDEX = "captions.index"
CAPTIONS_META = "captions_meta.json"
CAPTIONS_SBERT_INDEX = "captions_sbert.index"
CAPTIONS_SBERT_META = "captions_sbert.json"
PHOTOS_DIR = "photos"
THUMBS_DIR = "thumbs"

META_VERSION = 3
# v2 отличается от v3 только отсутствием подписи у снимка, а недостающие поля
# закрыты значениями по умолчанию — поэтому старые базы читаются как есть, без
# миграции и без перестроения индекса.
SUPPORTED_META_VERSIONS = (2, 3)
CAPTIONS_SBERT_VERSION = 1
# Вес пути CLIP при слиянии — см. docs/CAPTION_SEARCH.md.
#
# В C0 на человеческих подписях оптимум был 0,6, но работать система будет на
# машинных, а они слабее, и вес честнее брать оттуда: на подписях BLIP оптимум
# сместился к 0,7, и разница не косметическая — при 0,6 Recall@5 ниже на 1,8 п.п.
# Подбор в обоих замерах шёл на одной половине запросов, отчёт — на другой.
CAPTION_FUSION_ALPHA = 0.7
# Ниже этого покрытия слияние не включается.
#
# Оценки подписей приводятся к нулевому среднему по подписанным снимкам, и при
# малом покрытии это фабрикует сигнал из ничего: лучшая из трёх подписей получает
# высокую оценку просто потому, что она лучшая из трёх, а не потому, что отвечает
# запросу. Такой снимок вылезает наверх, вытесняя честные находки CLIP.
#
# Пока база размечена меньше чем наполовину, поиск остаётся обычным. Это заметно
# лучше, чем портить выдачу на всё время разметки.
CAPTION_FUSION_MIN_COVERAGE = 0.5
CAPTION_FUSION_MIN_PHOTOS = 10
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE = 320
# Индекс сбрасывается на диск не после каждого батча: write_index на 5000 векторов — это
# 10 МБ записи, делать это полтораста раз подряд при импорте архива бессмысленно.
FLUSH_EVERY = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Запрос -> вектор в пространстве подписей. Именно функция, а не модель: ядру
# незачем знать, чем считается текстовый эмбеддинг.
CaptionEncoderLike = Callable[[str], np.ndarray]


# --------------------------------------------------------------------------
# Типы данных
# --------------------------------------------------------------------------

@dataclass
class Photo:
    photo_id: str
    filename: str
    bytes: int = 0
    added_at: str = ""
    caption: str = ""
    caption_model: str = ""  # чем сгенерирована — чтобы уметь перегенерировать

    def to_dict(self) -> dict:
        data = {
            "photo_id": self.photo_id,
            "filename": self.filename,
            "bytes": self.bytes,
            "added_at": self.added_at,
        }
        # У базы без подписей meta остаётся ровно такой же, как была в v2: пустые
        # поля в каждой из пяти тысяч записей — это лишние сотни килобайт и шум
        # в диффе на ровном месте.
        if self.caption:
            data["caption"] = self.caption
        if self.caption_model:
            data["caption_model"] = self.caption_model
        return data


@dataclass
class SearchHit:
    photo_id: str
    score: float
    filename: str
    path: str
    caption: str = ""  # чтобы показать человеку, почему снимок нашёлся


@dataclass
class CaptionHit:
    photo_id: str
    score: float
    caption: str


@dataclass
class AddResult:
    added: list[Photo] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (имя файла, причина)

    @property
    def added_count(self) -> int:
        return len(self.added)


@dataclass
class Stats:
    photos_count: int
    photos_bytes: int
    index_bytes: int
    has_captions: bool
    captions_count: int = 0  # сколько снимков уже с подписью (покрытие неполное)


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_write_index(path: Path, index) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _file_photo_id(path: Path) -> str:
    """
    photo_id = первые 16 символов sha256 от содержимого файла. Идентификатор по содержимому
    даёт дедупликацию бесплатно: один и тот же снимок, загруженный дважды (в том числе под
    другим именем), не попадёт в базу повторно.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _zscore(values: np.ndarray) -> np.ndarray:
    """Нулевое среднее, единичный разброс. Константный вектор даёт нули, а не NaN."""
    return (values - values.mean()) / (values.std() + 1e-9)


def normalize_id(value) -> str:
    """
    Приводит числовой id к виду без ведущих нулей, чтобы '000000000139' и '139' считались
    одним image_id. Нечисловые id возвращаются как есть. Нужно для legacy-баз COCO.
    """
    s = str(value)
    return str(int(s)) if s.isdigit() else s


# --------------------------------------------------------------------------
# IndexStore
# --------------------------------------------------------------------------

class IndexStore:
    """
    Одна база. Экземпляр не потокобезопасен «сам по себе» — все мутирующие операции
    берут внутренний лок, поэтому его можно шарить между потоками (web + фоновая задача).
    """

    def __init__(self, root: Path, index, photos: list[Photo], legacy: bool = False):
        self.root = Path(root)
        self._index = index
        self._photos = photos
        self._legacy = legacy  # старый формат meta: [{"image_id","path"}], пути наружу базы
        self._lock = threading.RLock()
        self._captions_index = None
        self._captions_meta: list[dict] = []
        self._captions_loaded = False
        self._sbert_index = None
        self._sbert_rows: list[str] = []  # строка индекса -> photo_id
        self._sbert_model = ""
        self._sbert_loaded = False
        self._by_id = {p.photo_id: i for i, p in enumerate(photos)}

    # ------------------------------------------------------------------
    # Открытие / создание
    # ------------------------------------------------------------------

    @classmethod
    def create_empty(cls, root: str | Path, holder: ModelHolder | None = None) -> "IndexStore":
        """
        Создаёт пустую базу. Именно этого не умел прежний код: SearchEngine падал, если
        images.index не существует, поэтому «создать новую базу» было невозможно.
        """
        root = Path(root)
        if (root / IMAGES_INDEX).exists():
            raise StoreError(f"База в {root} уже существует")
        root.mkdir(parents=True, exist_ok=True)
        (root / PHOTOS_DIR).mkdir(exist_ok=True)
        (root / THUMBS_DIR).mkdir(exist_ok=True)

        holder = holder or ModelHolder.get()
        index = faiss.IndexFlatIP(holder.dim)
        store = cls(root, index, [])
        store._persist()
        return store

    @classmethod
    def open(cls, root: str | Path) -> "IndexStore":
        root = Path(root)
        index_path = root / IMAGES_INDEX
        meta_path = root / IMAGES_META
        if not index_path.exists() or not meta_path.exists():
            raise StoreError(f"В {root} нет базы (нужны {IMAGES_INDEX} и {IMAGES_META})")

        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        photos, legacy = cls._parse_meta(raw)
        if len(photos) != index.ntotal:
            raise StoreError(
                f"База {root} повреждена: {len(photos)} записей в meta против "
                f"{index.ntotal} векторов в индексе"
            )
        return cls(root, index, photos, legacy=legacy)

    @classmethod
    def open_or_create(cls, root: str | Path, holder: ModelHolder | None = None) -> "IndexStore":
        root = Path(root)
        if (root / IMAGES_INDEX).exists():
            return cls.open(root)
        return cls.create_empty(root, holder=holder)

    @staticmethod
    def _parse_meta(raw) -> tuple[list[Photo], bool]:
        """
        Читает все три формата meta:
          v3 (текущий): к полям v2 добавлены "caption" и "caption_model"
          v2:           {"version": 2, "photos": [{"photo_id","filename","bytes","added_at"}]}
          legacy:       [{"image_id": "...", "path": "data/images/..."}]
        Legacy-формат нужен, чтобы веб мог открыть уже построенную базу COCO из index/.
        """
        if isinstance(raw, dict) and raw.get("version") in SUPPORTED_META_VERSIONS:
            return [Photo(**item) for item in raw["photos"]], False

        if isinstance(raw, list):
            photos = [
                Photo(
                    photo_id=normalize_id(item["image_id"]),
                    filename=str(item["path"]).replace("\\", "/"),
                )
                for item in raw
            ]
            return photos, True

        raise StoreError("Неизвестный формат images_meta.json")

    # ------------------------------------------------------------------
    # Пути к файлам
    # ------------------------------------------------------------------

    def photo_path(self, photo: Photo | str) -> Path:
        """Абсолютный путь к оригиналу."""
        photo = self._resolve(photo)
        if self._legacy:
            # в старом формате в meta лежит путь относительно корня проекта
            p = Path(photo.filename)
            return p if p.is_absolute() else PROJECT_ROOT / p
        return self.root / PHOTOS_DIR / photo.filename

    def thumb_path(self, photo: Photo | str) -> Path | None:
        """Путь к превью или None, если превью нет (legacy-база превью не имеет)."""
        photo = self._resolve(photo)
        if self._legacy:
            return None
        path = self.root / THUMBS_DIR / f"{photo.photo_id}.webp"
        return path if path.exists() else None

    def _resolve(self, photo: Photo | str) -> Photo:
        if isinstance(photo, Photo):
            return photo
        idx = self._by_id.get(photo)
        if idx is None:
            raise KeyError(f"Фото {photo} нет в базе")
        return self._photos[idx]

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._photos)

    def has(self, photo_id: str) -> bool:
        return photo_id in self._by_id

    def get_photo(self, photo_id: str) -> Photo | None:
        index = self._by_id.get(photo_id)
        return self._photos[index] if index is not None else None

    def list_photos(self, offset: int = 0, limit: int | None = None) -> list[Photo]:
        """Свежие сверху — так галерея в вебе показывает только что добавленное первым."""
        ordered = list(reversed(self._photos))
        return ordered[offset:] if limit is None else ordered[offset:offset + limit]

    def stats(self) -> Stats:
        index_bytes = sum(
            (self.root / name).stat().st_size
            for name in (IMAGES_INDEX, IMAGES_META, CAPTIONS_INDEX, CAPTIONS_META,
                         CAPTIONS_SBERT_INDEX, CAPTIONS_SBERT_META)
            if (self.root / name).exists()
        )
        if self._legacy:
            # у legacy-базы размеров в meta нет — считаем по файлам (медленно, но разово)
            photos_bytes = 0
            for photo in self._photos:
                try:
                    photos_bytes += self.photo_path(photo).stat().st_size
                except OSError:
                    pass
        else:
            photos_bytes = sum(p.bytes for p in self._photos)
        return Stats(
            photos_count=len(self._photos),
            photos_bytes=photos_bytes,
            index_bytes=index_bytes,
            has_captions=self._load_captions() is not None,
            captions_count=sum(1 for p in self._photos if p.caption),
        )

    def iter_export_files(self) -> Iterator[tuple[str, Path]]:
        """(имя внутри архива, путь на диске) — для потоковой сборки zip при экспорте."""
        for photo in self._photos:
            path = self.photo_path(photo)
            if path.exists():
                yield f"photos/{path.name}", path

    def manifest(self) -> dict:
        return {
            "version": META_VERSION,
            "exported_at": _now(),
            "photos": [p.to_dict() for p in self._photos],
        }

    # ------------------------------------------------------------------
    # Добавление
    # ------------------------------------------------------------------

    def add_photos(
        self,
        files: Sequence[str | Path],
        *,
        on_progress: ProgressCallback | None = None,
        show_progress: bool = False,
        holder: ModelHolder | None = None,
    ) -> AddResult:
        """
        Пакетно добавляет файлы: валидация -> копирование в photos/ -> превью ->
        эмбеддинги батчами -> запись индекса.

        Битые и повторяющиеся файлы не роняют операцию, а попадают в result.skipped —
        при импорте архива на 1000 фото одна повреждённая картинка не должна отменять всё.
        """
        result = AddResult()
        with self._lock:
            if self._legacy:
                raise StoreError("Legacy-база открыта только для чтения")

            staged: list[tuple[Photo, Path]] = []
            seen: set[str] = set()

            for src in files:
                src = Path(src)
                name = src.name
                if src.suffix.lower() not in ALLOWED_SUFFIXES:
                    result.skipped.append((name, "неподдерживаемый формат"))
                    continue
                try:
                    photo_id = _file_photo_id(src)
                except OSError as e:
                    result.skipped.append((name, f"не читается: {e}"))
                    continue
                if photo_id in self._by_id or photo_id in seen:
                    result.skipped.append((name, "уже есть в базе"))
                    continue
                if not self._is_valid_image(src):
                    result.skipped.append((name, "не является изображением"))
                    continue

                seen.add(photo_id)
                dest = self.root / PHOTOS_DIR / f"{photo_id}{src.suffix.lower()}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                self._make_thumb(dest, self.root / THUMBS_DIR / f"{photo_id}.webp")
                staged.append(
                    (
                        Photo(
                            photo_id=photo_id,
                            filename=dest.name,
                            bytes=dest.stat().st_size,
                            added_at=_now(),
                        ),
                        dest,
                    )
                )

            if not staged:
                return result

            total = len(staged)
            processed = 0
            try:
                for start in range(0, total, FLUSH_EVERY):
                    chunk = staged[start:start + FLUSH_EVERY]
                    embeddings = compute_image_embeddings(
                        [path for _, path in chunk],
                        on_progress=(
                            (lambda done, _t, base=start: on_progress(base + done, total))
                            if on_progress
                            else None
                        ),
                        show_progress=show_progress,
                        holder=holder,
                    )
                    self._index.add(embeddings)
                    for photo, _ in chunk:
                        self._by_id[photo.photo_id] = len(self._photos)
                        self._photos.append(photo)
                        result.added.append(photo)
                    processed = start + len(chunk)
                    self._persist()  # чанк = FLUSH_EVERY, сброс идёт раз в 500 фото
            except BaseException:
                # прерывание на середине (отмена задачи, ошибка модели) не должно оставлять
                # скопированные, но не проиндексированные файлы: они не видны в базе,
                # зато занимают место и попадут в подсчёт объёма
                self._discard_staged(staged[processed:])
                self._persist()
                raise

            return result

    def _discard_staged(self, staged: list[tuple[Photo, Path]]) -> None:
        for photo, path in staged:
            for target in (path, self.root / THUMBS_DIR / f"{photo.photo_id}.webp"):
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _is_valid_image(path: Path) -> bool:
        """Расширение ничего не гарантирует — проверяем, что файл действительно картинка."""
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @staticmethod
    def _make_thumb(src: Path, dest: Path) -> None:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.save(dest, "WEBP", quality=80)
        except Exception as e:  # превью — не повод отменять добавление фото
            print(f"[превью не создано] {src.name}: {e}")

    # ------------------------------------------------------------------
    # Удаление
    # ------------------------------------------------------------------

    def delete_photos(self, photo_ids: Sequence[str]) -> int:
        """
        Удаляет фото из индекса, метаданных и с диска. Возвращает количество удалённых.

        Пересчёт эмбеддингов не нужен: IndexFlatIP хранит сырые векторы, поэтому индекс
        пересобирается из reconstruct_n за миллисекунды (5000x512 float32 ≈ 10 МБ).
        """
        with self._lock:
            if self._legacy:
                raise StoreError("Legacy-база открыта только для чтения")

            targets = {pid for pid in photo_ids if pid in self._by_id}
            if not targets:
                return 0

            keep_rows = [i for i, p in enumerate(self._photos) if p.photo_id not in targets]
            removed = [p for p in self._photos if p.photo_id in targets]

            self._index = self._rebuild_index(self._index, keep_rows)
            self._photos = [self._photos[i] for i in keep_rows]
            self._by_id = {p.photo_id: i for i, p in enumerate(self._photos)}

            self._drop_captions_of(targets)
            self._drop_caption_vectors_of(targets)
            self._persist()

            # файлы удаляем после успешной записи индекса: если процесс упадёт раньше,
            # база останется консистентной, максимум — осиротеют файлы на диске
            for photo in removed:
                for path in (self.root / PHOTOS_DIR / photo.filename,
                             self.root / THUMBS_DIR / f"{photo.photo_id}.webp"):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as e:
                        print(f"[файл не удалён] {path}: {e}")

            return len(removed)

    @staticmethod
    def _rebuild_index(index, keep_rows: list[int]):
        """Новый IndexFlatIP только из строк keep_rows исходного индекса."""
        new_index = faiss.IndexFlatIP(index.d)
        if keep_rows:
            vectors = index.reconstruct_n(0, index.ntotal)
            new_index.add(np.ascontiguousarray(vectors[keep_rows], dtype="float32"))
        return new_index

    def _drop_captions_of(self, photo_ids: set[str]) -> None:
        """Подписи удалённых картинок тоже уходят — иначе поиск фото->текст врёт."""
        if self._load_captions() is None:
            return
        keep_rows = [
            i for i, item in enumerate(self._captions_meta)
            if normalize_id(item["image_id"]) not in photo_ids
        ]
        if len(keep_rows) == len(self._captions_meta):
            return
        self._captions_index = self._rebuild_index(self._captions_index, keep_rows)
        self._captions_meta = [self._captions_meta[i] for i in keep_rows]
        _atomic_write_index(self.root / CAPTIONS_INDEX, self._captions_index)
        _atomic_write_json(self.root / CAPTIONS_META, self._captions_meta)

    # ------------------------------------------------------------------
    # Подписи
    # ------------------------------------------------------------------
    #
    # Разделены намеренно: текст пишет генератор подписей (BLIP), векторы —
    # текстовая модель (SBERT). Core не знает ни о том, ни о другом и принимает
    # готовые значения, поэтому ни веб, ни бот не тянут лишних зависимостей.

    def set_caption_texts(self, captions: dict[str, str], *, model: str = "") -> int:
        """
        Проставляет подписи снимкам. Неизвестные photo_id молча пропускаются:
        генератор работает в фоне, и снимок мог быть удалён, пока его подпись
        считалась, — это штатный ход событий, а не ошибка.
        """
        with self._lock:
            if self._legacy:
                raise StoreError("Legacy-база открыта только для чтения")
            changed = 0
            for photo_id, caption in captions.items():
                row = self._by_id.get(photo_id)
                if row is None:
                    continue
                self._photos[row].caption = caption
                self._photos[row].caption_model = model
                changed += 1
            if changed:
                self._persist()
            return changed

    def caption_of(self, photo_id: str) -> str:
        photo = self.get_photo(photo_id)
        return photo.caption if photo else ""

    def photos_without_caption(self, limit: int | None = None) -> list[Photo]:
        """Очередь работы для генератора подписей."""
        pending = [p for p in self._photos if not p.caption]
        return pending if limit is None else pending[:limit]

    def captions_coverage(self) -> tuple[int, int]:
        """(снимков с подписью, всего снимков)."""
        return sum(1 for p in self._photos if p.caption), len(self._photos)

    def set_caption_vectors(
        self, vectors: dict[str, np.ndarray], *, model: str = ""
    ) -> int:
        """
        Добавляет или заменяет векторы подписей. Индекс собирается заново из
        объединения старых и новых строк: IndexFlatIP не умеет заменять вектор на
        месте, а размер здесь мал (одна строка на снимок, а не пять, как у COCO).

        Смена модели распознаётся по размерности: векторы другой длины означают
        другой энкодер, и старые строки в таком случае выбрасываются — смешивать
        два пространства в одном индексе нельзя, оценки были бы бессмысленны.
        """
        with self._lock:
            if self._legacy:
                raise StoreError("Legacy-база открыта только для чтения")

            fresh = {
                photo_id: np.asarray(vector, dtype="float32").reshape(-1)
                for photo_id, vector in vectors.items()
                if photo_id in self._by_id
            }
            if not fresh:
                return 0

            dims = {vector.shape[0] for vector in fresh.values()}
            if len(dims) != 1:
                raise StoreError(f"Векторы подписей разной размерности: {sorted(dims)}")
            dim = dims.pop()

            merged: dict[str, np.ndarray] = {}
            self._load_caption_vectors()
            if self._sbert_index is not None and self._sbert_index.d == dim:
                stored = self._sbert_index.reconstruct_n(0, self._sbert_index.ntotal)
                for row, photo_id in enumerate(self._sbert_rows):
                    if photo_id in self._by_id:
                        merged[photo_id] = stored[row]
            merged.update(fresh)

            index = faiss.IndexFlatIP(dim)
            rows = list(merged.keys())
            index.add(np.ascontiguousarray([merged[pid] for pid in rows], dtype="float32"))

            self._sbert_index = index
            self._sbert_rows = rows
            self._sbert_model = model or self._sbert_model
            self._sbert_loaded = True
            self._persist_caption_vectors()
            return len(fresh)

    def search_captions(self, query_vector: np.ndarray, top_k: int = 5) -> list[CaptionHit]:
        """
        Поиск по подписям. Вектор запроса приходит готовым — кодирует его тот, кто
        владеет текстовой моделью.

        Строки, чьи снимки уже удалены, пропускаются: индекс подписей вторичен и
        может отставать от базы, и это не повод отдавать наружу ссылки в никуда.
        """
        self._load_caption_vectors()
        if self._sbert_index is None:
            return []
        emb = np.ascontiguousarray(
            np.asarray(query_vector, dtype="float32").reshape(1, -1)
        )
        if emb.shape[1] != self._sbert_index.d:
            raise StoreError(
                f"Размерность запроса {emb.shape[1]} не совпадает с индексом "
                f"подписей {self._sbert_index.d} — модель сменилась?"
            )

        hits: list[CaptionHit] = []
        for row, score in self._search(self._sbert_index, emb, top_k, lambda r, s: (r, s)):
            photo_id = self._sbert_rows[row]
            photo = self.get_photo(photo_id)
            if photo is None:
                continue
            hits.append(CaptionHit(photo_id=photo_id, score=score, caption=photo.caption))
        return hits

    def caption_index_model(self) -> str:
        self._load_caption_vectors()
        return self._sbert_model

    def _load_caption_vectors(self):
        """
        Лениво читает индекс подписей. Расхождение числа векторов с собственным
        списком строк означает, что индекс испорчен, — тогда поиск по подписям
        просто выключается. Ронять базу целиком из-за вторичного индекса нельзя:
        снимки и поиск по ним от этого не страдают.
        """
        if self._sbert_loaded:
            return self._sbert_index
        self._sbert_loaded = True

        index_path = self.root / CAPTIONS_SBERT_INDEX
        meta_path = self.root / CAPTIONS_SBERT_META
        if not (index_path.exists() and meta_path.exists()):
            return None
        try:
            index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rows = list(meta["rows"])
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            print(f"[индекс подписей не прочитан] {self.root}: {e}")
            return None

        if index.ntotal != len(rows):
            print(
                f"[индекс подписей рассогласован] {self.root}: {index.ntotal} векторов "
                f"против {len(rows)} строк — поиск по подписям выключен"
            )
            return None

        self._sbert_index = index
        self._sbert_rows = rows
        self._sbert_model = meta.get("model", "")
        return self._sbert_index

    def _persist_caption_vectors(self) -> None:
        _atomic_write_index(self.root / CAPTIONS_SBERT_INDEX, self._sbert_index)
        _atomic_write_json(
            self.root / CAPTIONS_SBERT_META,
            {
                "version": CAPTIONS_SBERT_VERSION,
                "model": self._sbert_model,
                "rows": self._sbert_rows,
            },
        )

    def _drop_caption_vectors_of(self, photo_ids: set[str]) -> None:
        """
        Выбрасывает векторы удалённых снимков.

        Здесь и видно, зачем индексу подписей собственный список photo_id: удаляют
        обычно снимок, у которого подписи ещё нет, и при позиционной адресации все
        последующие строки уехали бы на единицу, тихо переклеив подписи на чужие
        фотографии.
        """
        if self._load_caption_vectors() is None:
            return
        keep = [row for row, photo_id in enumerate(self._sbert_rows)
                if photo_id not in photo_ids]
        if len(keep) == len(self._sbert_rows):
            return
        self._sbert_index = self._rebuild_index(self._sbert_index, keep)
        self._sbert_rows = [self._sbert_rows[row] for row in keep]
        self._persist_caption_vectors()

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        translate: bool = True,
        holder: ModelHolder | None = None,
        caption_encoder: "CaptionEncoderLike | None" = None,
        alpha: float = CAPTION_FUSION_ALPHA,
    ) -> tuple[str, list[SearchHit]]:
        """
        Возвращает (использованный запрос, результаты). Пустая база -> пустой список.

        caption_encoder — то, чем кодировать запрос для поиска по подписям. Передаётся
        функцией, а не моделью: ядро не должно зависеть от конкретного текстового
        энкодера. Если его нет или в базе нет ни одной подписи, работает обычный CLIP.

        Кодируется именно used_query, то есть уже переведённый текст: подписи
        генерируются на английском, и слать в текстовую модель русский оригинал
        значило бы сравнивать разные языки.
        """
        holder = holder or ModelHolder.get()
        used_query = maybe_translate(query, self.root / TRANSLATE_CACHE_FILE, translate)
        emb = holder.encode_texts([used_query])

        if caption_encoder is not None and self.fusion_ready():
            return used_query, self._search_fused(emb, caption_encoder(used_query),
                                                  top_k, alpha)
        return used_query, self._search(self._index, emb, top_k, self._hit_from_photo)

    def fusion_ready(self) -> bool:
        """
        Хватает ли подписей, чтобы слияние помогало, а не мешало. Пороги и причина —
        рядом с CAPTION_FUSION_MIN_COVERAGE.
        """
        if self._load_caption_vectors() is None or not self._photos:
            return False
        covered = self._sbert_index.ntotal
        return (
            covered >= CAPTION_FUSION_MIN_PHOTOS
            and covered / len(self._photos) >= CAPTION_FUSION_MIN_COVERAGE
        )

    def _search_fused(
        self, emb: np.ndarray, caption_vector: np.ndarray, top_k: int, alpha: float
    ) -> list[SearchHit]:
        """
        Слияние двух путей поиска — ровно то, что мерилось в C0.

        Там оценки приводились к нулевому среднему и единичному разбросу по ВСЕМ
        кандидатам, и повторить это надо в точности: у CLIP косинусы плотно сидят
        около 0.2-0.3, у текстовой модели разброс шире, и сложение сырых оценок
        отдало бы вес просто тому, у кого шкала растянутее. Поэтому берём у FAISS
        полную выдачу, а не top_k: нормировать по верхушке списка нельзя, среднее
        и разброс по ней — не те же самые числа.

        Снимок без подписи получает по второму пути ноль, то есть ровно среднее.
        Это нейтральная подстановка: отсутствие подписи не повышает и не понижает
        снимок, а оставляет его на том месте, куда его поставил CLIP. Штрафовать
        такие снимки было бы неверно — база размечается постепенно, и в середине
        разметки половина фотографий просто ещё не дошла до очереди.
        """
        with self._lock:
            total = self._index.ntotal
            if total == 0 or top_k <= 0:
                return []

            clip_scores = self._full_scores(self._index, emb, total)

            caption_z = np.zeros(total, dtype="float32")
            vector = np.ascontiguousarray(
                np.asarray(caption_vector, dtype="float32").reshape(1, -1)
            )
            if vector.shape[1] != self._sbert_index.d:
                raise StoreError(
                    f"Размерность запроса {vector.shape[1]} не совпадает с индексом "
                    f"подписей {self._sbert_index.d} — модель сменилась?"
                )
            raw = self._full_scores(self._sbert_index, vector, self._sbert_index.ntotal)
            raw_z = _zscore(raw)
            for row, photo_id in enumerate(self._sbert_rows):
                position = self._by_id.get(photo_id)
                if position is not None:
                    caption_z[position] = raw_z[row]

            fused = alpha * _zscore(clip_scores) + (1 - alpha) * caption_z
            best = np.argsort(-fused)[:top_k]
            return [self._hit_from_photo(int(row), float(fused[row])) for row in best]

    @staticmethod
    def _full_scores(index, emb: np.ndarray, total: int) -> np.ndarray:
        """
        Оценки по всем векторам индекса в порядке строк. FAISS всё равно сканирует
        индекс целиком, поэтому полная выдача вместо top_k стоит почти столько же:
        на индексе COCO (5022 вектора, dim 512) замерено 1,30 мс против 0,84 мс.

        На фоне запроса это ничто — время уходит в энкодеры, а не в поиск. Дороже
        обходится сам второй путь: подпись запроса надо закодировать текстовой
        моделью, и это единственная заметная добавка слияния.
        """
        scores, rows = index.search(emb, total)
        full = np.zeros(total, dtype="float32")
        full[rows[0]] = scores[0]
        return full

    def search_similar(self, photo_id: str, top_k: int = 5) -> list[SearchHit]:
        """
        Похожие на снимок, который уже лежит в базе.

        Эмбеддинг заново не считается: вектор снимка уже есть в индексе, и
        reconstruct достаёт его обратно. Для сценария «прислал фото — покажи
        похожие» это убирает второй прогон энкодера, то есть примерно половину
        всей работы.

        Сам снимок из выдачи исключается: показывать человеку его же фотографию
        как «самую похожую» бессмысленно, а первое место она займёт всегда.
        """
        with self._lock:
            row = self._by_id.get(photo_id)
            if row is None or self._index.ntotal == 0:
                return []
            vector = np.ascontiguousarray(
                self._index.reconstruct(row).reshape(1, -1), dtype="float32"
            )

        # берём на один больше, чтобы после выброса самого снимка осталось top_k
        hits = self._search(self._index, vector, top_k + 1, self._hit_from_photo)
        return [hit for hit in hits if hit.photo_id != photo_id][:top_k]

    def search_image(
        self,
        image: Image.Image | str | Path,
        top_k: int = 5,
        holder: ModelHolder | None = None,
    ) -> tuple[list[SearchHit], list[CaptionHit]]:
        """Похожие картинки и (если у базы есть индекс подписей) релевантные подписи."""
        holder = holder or ModelHolder.get()
        if not isinstance(image, Image.Image):
            image = open_image(image)
        emb = holder.encode_images([image])

        hits = self._search(self._index, emb, top_k, self._hit_from_photo)

        captions: list[CaptionHit] = []
        if self._load_captions() is not None:
            captions = self._search(
                self._captions_index, emb, top_k,
                lambda row, score: CaptionHit(
                    photo_id=normalize_id(self._captions_meta[row]["image_id"]),
                    score=score,
                    caption=self._captions_meta[row]["caption"],
                ),
            )
        return hits, captions

    def _search(self, index, emb: np.ndarray, top_k: int, build):
        """
        Общая обёртка над index.search. Главное здесь — фильтр row < 0: у пустой или
        почти пустой базы FAISS возвращает -1 в незаполненных позициях, и прежний код
        на этом падал бы, обращаясь к meta[-1].
        """
        if index is None or index.ntotal == 0 or top_k <= 0:
            return []
        with self._lock:
            scores, rows = index.search(emb, min(top_k, index.ntotal))
        return [build(int(row), float(score))
                for row, score in zip(rows[0], scores[0]) if row >= 0]

    def _hit_from_photo(self, row: int, score: float) -> SearchHit:
        photo = self._photos[row]
        return SearchHit(
            photo_id=photo.photo_id,
            score=score,
            filename=photo.filename,
            path=str(self.photo_path(photo)),
            caption=photo.caption,
        )

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """
        Атомарная запись индекса и meta: сначала .tmp, затем os.replace.

        Версия проставляется по факту: пока в базе нет ни одной подписи, файл
        остаётся ровно таким же, как писала предыдущая версия кода. Это не
        косметика — данные общие между локальным запуском и контейнерами, и
        собранный ранее образ на meta v3 просто не откроет базу. Так окно
        несовместимости появляется только у тех баз, где подписи и правда есть.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        version = META_VERSION if any(p.caption for p in self._photos) else 2
        _atomic_write_index(self.root / IMAGES_INDEX, self._index)
        _atomic_write_json(
            self.root / IMAGES_META,
            {"version": version, "photos": [p.to_dict() for p in self._photos]},
        )

    def _load_captions(self):
        """Индекс подписей загружается лениво: у пользовательских баз его обычно нет."""
        if not self._captions_loaded:
            self._captions_loaded = True
            index_path = self.root / CAPTIONS_INDEX
            meta_path = self.root / CAPTIONS_META
            if index_path.exists() and meta_path.exists():
                self._captions_index = faiss.read_index(str(index_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._captions_meta = json.load(f)
        return self._captions_index
