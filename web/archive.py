"""
Приём zip-архивов от пользователя и сборка zip на экспорт.

Это самая опасная точка приложения: содержимое архива полностью контролируется тем, кто
его загрузил. Проверки ниже — не паранойя, а закрытие конкретных известных атак
(docs/WEB_PLAN.md, раздел 7.2):

* zip-slip — имя члена архива вида "../../etc/passwd" или "C:\\Windows\\..." заставляет
  extractall писать за пределы целевой папки. Поэтому extractall не используется вовсе:
  из каждого имени берётся только basename.
* zip-бомба — архив в мегабайт, разворачивающийся в терабайты. Поэтому распакованный
  размер суммируется ДО распаковки и сверяется с лимитом.
* symlink-члены — распакованная ссылка может указывать куда угодно, и следующая запись
  «в неё» уйдёт мимо песочницы. Такие члены пропускаются.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
S_IFLNK = 0o120000


class ArchiveError(ValueError):
    """Архив не принят. Текст предназначен для показа пользователю."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 20_000
    max_total_bytes: int = 3 * 1024 ** 3
    max_member_bytes: int = 100 * 1024 ** 2


@dataclass
class ExtractResult:
    files: list[Path]
    skipped: list[tuple[str, str]]  # (имя в архиве, причина)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == S_IFLNK


def _safe_name(raw_name: str) -> str | None:
    """
    Превращает имя члена архива в безопасное имя файла или возвращает None.

    Структура папок внутри архива не сохраняется намеренно: базе она не нужна (фото
    адресуются по photo_id), а любое её сохранение — это ещё один шанс промахнуться
    мимо песочницы.
    """
    normalized = raw_name.replace("\\", "/")
    if normalized.endswith("/"):
        return None  # запись о папке
    name = Path(normalized).name
    if not name or name in (".", ".."):
        return None
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    return name


def inspect(zip_path: Path, limits: ArchiveLimits) -> tuple[int, int]:
    """
    Проверяет архив ДО распаковки. Возвращает (количество картинок, суммарный размер).
    Бросает ArchiveError, если архив битый или превышает лимиты.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ArchiveError("Это не zip-архив")

    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"Архив повреждён: {e}") from e

    if len(infos) > limits.max_members:
        raise ArchiveError(
            f"В архиве слишком много файлов ({len(infos)}), лимит {limits.max_members}"
        )

    count, total = 0, 0
    for info in infos:
        if _is_symlink(info) or _safe_name(info.filename) is None:
            continue
        if info.file_size > limits.max_member_bytes:
            raise ArchiveError(
                f"Файл {Path(info.filename).name} слишком большой "
                f"({info.file_size // 1024 // 1024} МБ)"
            )
        count += 1
        total += info.file_size
        # сумма считается на лету: узнать о превышении надо до того, как на диск
        # записан хоть один байт
        if total > limits.max_total_bytes:
            raise ArchiveError(
                f"Распакованный размер превышает лимит "
                f"({limits.max_total_bytes // 1024 // 1024} МБ)"
            )

    if count == 0:
        raise ArchiveError("В архиве нет изображений (jpg, jpeg, png, webp)")
    return count, total


def extract_images(zip_path: Path, dest_dir: Path, limits: ArchiveLimits) -> ExtractResult:
    """
    Распаковывает только изображения и только в dest_dir, плоским списком.
    Вызывать после inspect() — она проверяет лимиты и осмысленность архива.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    skipped: list[tuple[str, str]] = []
    used_names: set[str] = set()
    written = 0

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if _is_symlink(info):
                skipped.append((info.filename, "символическая ссылка"))
                continue
            name = _safe_name(info.filename)
            if name is None:
                if not info.filename.endswith("/"):
                    skipped.append((info.filename, "не изображение"))
                continue
            if info.file_size > limits.max_member_bytes:
                skipped.append((info.filename, "слишком большой файл"))
                continue

            written += info.file_size
            if written > limits.max_total_bytes:
                raise ArchiveError("Распакованный размер превышает лимит")

            target = dest_dir / _unique_name(name, used_names)
            # copyfileobj по потоку, а не read() целиком: файл может быть на 100 МБ
            with archive.open(info) as source, open(target, "wb") as out:
                _copy_limited(source, out, limits.max_member_bytes)
            files.append(target)

    return ExtractResult(files=files, skipped=skipped)


def _unique_name(name: str, used: set[str]) -> str:
    """Разные папки архива могут содержать одинаковые имена — разводим их суффиксом."""
    if name not in used:
        used.add(name)
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(1, 10_000):
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ArchiveError(f"Слишком много файлов с именем {name}")


def _copy_limited(source, target, limit: int, chunk_size: int = 1 << 20) -> None:
    """
    Копирует с ограничением: заголовок zip может врать о размере, поэтому доверять
    info.file_size на этапе записи нельзя — считаем фактически прочитанное.
    """
    written = 0
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return
        written += len(chunk)
        if written > limit:
            raise ArchiveError("Файл в архиве больше, чем заявлено в его заголовке")
        target.write(chunk)


class _ZipSink(io.RawIOBase):
    """
    Приёмник байтов для zipfile, из которого их забирают порциями.

    Наивная реализация «писать в BytesIO и после каждого файла очищать его» ломает
    архив: zipfile запоминает смещения записей через fp.tell(), а очистка буфера
    сбрасывает позицию в ноль — центральный каталог начинает указывать не туда,
    и такой архив не открывается. Поэтому позиция здесь считается независимо от
    того, сколько байт уже отдано наружу.

    seekable() = False — для zipfile это сигнал писать в потоковом режиме, без
    возврата назад для правки заголовков.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._position = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def write(self, data) -> int:
        data = bytes(data)
        self._chunks.append(data)
        self._position += len(data)
        return len(data)

    def tell(self) -> int:
        return self._position

    def take(self) -> bytes:
        """Забирает накопленное, не трогая счётчик позиции."""
        if not self._chunks:
            return b""
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


def stream_zip(entries: Iterator[tuple[str, Path]], manifest: bytes | None = None,
               chunk_size: int = 1 << 20):
    """
    Генератор байтов zip-архива для потоковой отдачи.

    ZIP_STORED без сжатия: фотографии — уже сжатый JPEG, тратить на них CPU бессмысленно.
    Зато без сжатия архив отдаётся на лету, не собираясь целиком ни в памяти, ни во
    временном файле: база на 2 ГБ иначе потребовала бы 2 ГБ свободного места только
    ради того, чтобы её скачать.
    """
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED) as archive:
        if manifest is not None:
            archive.writestr("manifest.json", manifest)
            yield sink.take()

        for arcname, path in entries:
            try:
                info = zipfile.ZipInfo.from_file(path, arcname)
            except OSError:
                # файл удалили, пока шла выгрузка, — пропускаем: обрывать скачивание
                # целой базы из-за одного снимка неправильно
                continue
            info.compress_type = zipfile.ZIP_STORED
            try:
                with archive.open(info, "w") as target, open(path, "rb") as source:
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
                        # отдаём порциями прямо во время записи файла: иначе стомегабайтный
                        # снимок целиком осел бы в памяти
                        yield sink.take()
            except OSError:
                continue
            yield sink.take()

    yield sink.take()  # центральный каталог, дописанный при закрытии
