"""
IndexStore — одна база фотографий: FAISS-индекс + файлы + метаданные.

В отличие от прежнего SearchEngine (bot/inference.py), который умел только открыть готовый
индекс и добавить в него по одной картинке, IndexStore закрывает весь пользовательский
сценарий веб-интерфейса: создать пустую базу, пакетно добавить, удалить, посчитать объём,
отдать файлы на экспорт.

Раскладка папки базы:
    <root>/
        images.index          FAISS IndexFlatIP по эмбеддингам картинок
        images_meta.json      {"version": 2, "photos": [...]} — порядок = порядок векторов
        captions.index        опционально: эмбеддинги подписей (есть только у COCO-базы)
        captions_meta.json
        translate_cache.json
        photos/               оригиналы, имя файла = <photo_id><ext>
        thumbs/               превью 320px WebP

Инвариант, на котором держится всё остальное: i-я запись в images_meta.json описывает
i-й вектор в images.index. Любая операция, меняющая одно, обязана менять и второе — под
локом и с атомарной записью на диск.
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
from typing import Iterator, Sequence

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
PHOTOS_DIR = "photos"
THUMBS_DIR = "thumbs"

META_VERSION = 2
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE = 320
# Индекс сбрасывается на диск не после каждого батча: write_index на 5000 векторов — это
# 10 МБ записи, делать это полтораста раз подряд при импорте архива бессмысленно.
FLUSH_EVERY = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Типы данных
# --------------------------------------------------------------------------

@dataclass
class Photo:
    photo_id: str
    filename: str
    bytes: int = 0
    added_at: str = ""

    def to_dict(self) -> dict:
        return {
            "photo_id": self.photo_id,
            "filename": self.filename,
            "bytes": self.bytes,
            "added_at": self.added_at,
        }


@dataclass
class SearchHit:
    photo_id: str
    score: float
    filename: str
    path: str


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
        Читает оба формата meta:
          v2 (новый):  {"version": 2, "photos": [{"photo_id","filename","bytes","added_at"}]}
          legacy:      [{"image_id": "...", "path": "data/images/..."}]
        Legacy-формат нужен, чтобы веб мог открыть уже построенную базу COCO из index/.
        """
        if isinstance(raw, dict) and raw.get("version") == META_VERSION:
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
            for name in (IMAGES_INDEX, IMAGES_META, CAPTIONS_INDEX, CAPTIONS_META)
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
    # Поиск
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        translate: bool = True,
        holder: ModelHolder | None = None,
    ) -> tuple[str, list[SearchHit]]:
        """Возвращает (использованный запрос, результаты). Пустая база -> пустой список."""
        holder = holder or ModelHolder.get()
        used_query = maybe_translate(query, self.root / TRANSLATE_CACHE_FILE, translate)
        emb = holder.encode_texts([used_query])
        return used_query, self._search(self._index, emb, top_k, self._hit_from_photo)

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
        )

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Атомарная запись индекса и meta: сначала .tmp, затем os.replace."""
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_index(self.root / IMAGES_INDEX, self._index)
        _atomic_write_json(
            self.root / IMAGES_META,
            {"version": META_VERSION, "photos": [p.to_dict() for p in self._photos]},
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
