"""
Добавление и удаление фотографий: загрузка файлов, импорт zip-архива.

Долгие операции уходят в очередь (web/jobs.py) и возвращают job_id: индексация тысячи
фото на CPU занимает десятки минут, столько ни один браузер соединение не держит.
Несколько файлов обрабатываются сразу — ради одной картинки гонять поллинг незачем.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from core.store import PROJECT_ROOT, IndexStore
from web import db
from web.archive import ArchiveError, ArchiveLimits, extract_images, inspect
from web.config import get_settings
from web.deps import OwnedDatabase, WritableDatabase
from web.jobs import JobContext, job_queue
from web.schemas import AddPhotosOut, DeletePhotosRequest, PhotoOut, PhotoPageOut
from web.stores import database_root, store_for, sync_stats

router = APIRouter(prefix="/api/databases/{database_id}", tags=["photos"])

# До этого числа файлов включительно работаем синхронно: ждать поллинга ради одного
# снимка — хуже, чем подождать секунду ответа.
SYNC_UPLOAD_LIMIT = 3
MAX_UPLOAD_FILES = 500
UPLOAD_CHUNK = 1 << 20


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------

def _limits_for(user_id: str) -> ArchiveLimits:
    """Лимит на архив — это остаток дисковой квоты пользователя, а не отдельное число."""
    settings = get_settings()
    remaining = max(0, settings.max_bytes_per_user - db.user_total_bytes(user_id))
    return ArchiveLimits(max_total_bytes=remaining)


def _check_room(database: dict, incoming: int) -> None:
    settings = get_settings()
    if database["photos_count"] + incoming > settings.max_photos_per_db:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"В базе не более {settings.max_photos_per_db} фото "
            f"(сейчас {database['photos_count']})",
        )
    if db.user_total_bytes(database["user_id"]) >= settings.max_bytes_per_user:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")


def _require_idle(database: dict) -> None:
    """
    Две одновременные операции над одной базой не имеют смысла: они всё равно
    выстроятся в очередь, но пользователь увидит два конкурирующих прогресс-бара.
    """
    if db.has_active_job(database["id"]):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "База уже обрабатывается, дождитесь окончания"
        )


def _tmp_dir(database: dict, name: str) -> Path:
    root = get_settings().database_dir(database["user_id"], database["id"]) / "tmp" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_upload(upload: UploadFile, target: Path, max_bytes: int) -> int:
    """Пишет загруженный файл на диск потоком, обрывая приём при превышении лимита."""
    written = 0
    with open(target, "wb") as out:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Файл слишком большой"
                )
            out.write(chunk)
    return written


def _index_files(store: IndexStore, files: list[Path], context: JobContext | None,
                 names: dict[str, str] | None = None) -> dict:
    """
    names — соответствие «имя во временной папке -> имя, которое дал файлу пользователь».
    Во временной папке имена снабжены порядковым префиксом (чтобы два файла с одинаковым
    именем не затёрли друг друга), но показывать этот префикс в списке пропущенных нельзя:
    пользователь не найдёт у себя «0001_photo.jpg».
    """
    result = store.add_photos(
        files,
        on_progress=(context.progress if context else None),
    )
    skipped = [((names or {}).get(name, name), reason) for name, reason in result.skipped]
    return {"added": result.added_count, "skipped": skipped}


# --------------------------------------------------------------------------
# Список фотографий
# --------------------------------------------------------------------------

@router.get("/photos", response_model=PhotoPageOut)
def list_photos(database: OwnedDatabase, offset: int = 0, limit: int = 60) -> PhotoPageOut:
    store = store_for(database)
    limit = max(1, min(limit, 200))
    photos = store.list_photos(offset=max(0, offset), limit=limit)
    return PhotoPageOut(
        total=len(store),
        offset=offset,
        items=[
            PhotoOut(photo_id=p.photo_id, bytes=p.bytes, added_at=p.added_at) for p in photos
        ],
    )


# --------------------------------------------------------------------------
# Отдача файлов
# --------------------------------------------------------------------------

def resolve_photo_file(database: dict, photo_id: str, *, thumb: bool) -> Path:
    """
    Путь к файлу фотографии с двойной проверкой.

    Первая — владение базой (её делает зависимость get_owned_database). Вторая —
    что итоговый путь физически лежит внутри папки этой базы: photo_id приходит
    от пользователя, и один этот факт обязывает не доверять ему при построении пути.
    """
    store = store_for(database)
    photo = store.get_photo(photo_id)
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")

    path = (store.thumb_path(photo) or store.photo_path(photo)) if thumb else store.photo_path(photo)

    allowed = [database_root(database["user_id"], database["id"], database.get("kind", "personal"))]
    if database.get("kind") == "demo":
        # У демо-базы (индекс COCO в старом формате) снимки лежат не внутри папки базы,
        # а рядом с ней — в data/images. Границей служит папка, содержащая индекс:
        # в рабочей установке это корень проекта, и за его пределы выйти всё равно нельзя.
        allowed.append(get_settings().demo_index_dir.parent)
        allowed.append(PROJECT_ROOT)

    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed) or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Файл недоступен")
    return path


def _photo_response(path: Path) -> FileResponse:
    # photo_id — это хеш содержимого, поэтому файл по такому адресу никогда не меняется
    # и его можно кэшировать вечно. private, а не public: содержимое принадлежит
    # конкретному пользователю и не должно оседать в общих прокси.
    return FileResponse(
        path, headers={"Cache-Control": "private, max-age=31536000, immutable"}
    )


@router.get("/photos/{photo_id}/thumb")
def get_thumb(database: OwnedDatabase, photo_id: str) -> FileResponse:
    return _photo_response(resolve_photo_file(database, photo_id, thumb=True))


@router.get("/photos/{photo_id}/file")
def get_photo_file(database: OwnedDatabase, photo_id: str) -> FileResponse:
    return _photo_response(resolve_photo_file(database, photo_id, thumb=False))


# --------------------------------------------------------------------------
# Добавление файлов
# --------------------------------------------------------------------------

@router.post("/photos", response_model=AddPhotosOut, status_code=status.HTTP_202_ACCEPTED)
def add_photos(database: WritableDatabase, files: list[UploadFile] = File(...)) -> AddPhotosOut:
    if not files:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Не переданы файлы")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"За раз не больше {MAX_UPLOAD_FILES} файлов",
        )
    _require_idle(database)
    _check_room(database, len(files))

    limits = _limits_for(database["user_id"])
    staging = _tmp_dir(database, f"upload-{db.new_id()}")
    saved: list[Path] = []
    names: dict[str, str] = {}
    try:
        for index, upload in enumerate(files):
            name = Path(upload.filename or f"file{index}").name or f"file{index}"
            target = staging / f"{index:04d}_{name}"
            _save_upload(upload, target, limits.max_member_bytes)
            saved.append(target)
            names[target.name] = name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if len(saved) <= SYNC_UPLOAD_LIMIT:
        try:
            store = store_for(database)
            outcome = _index_files(store, saved, None, names)
            sync_stats(database["id"], store)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return AddPhotosOut(added=outcome["added"], skipped=outcome["skipped"])

    def task(context: JobContext) -> str:
        try:
            store = store_for(database)
            outcome = _index_files(store, saved, context, names)
            return f"Добавлено фото: {outcome['added']}, пропущено: {len(outcome['skipped'])}"
        finally:
            # папку чистим и при отмене, и при ошибке: временные файлы уже скопированы
            # внутрь базы, второй экземпляр никому не нужен
            shutil.rmtree(staging, ignore_errors=True)
            store = store_for(database)
            sync_stats(database["id"], store)

    job = job_queue.submit(
        kind="add_photos",
        user_id=database["user_id"],
        database_id=database["id"],
        function=task,
        total=len(saved),
    )
    return AddPhotosOut(job_id=job["id"])


# --------------------------------------------------------------------------
# Импорт архива
# --------------------------------------------------------------------------

@router.post("/import", response_model=AddPhotosOut, status_code=status.HTTP_202_ACCEPTED)
def import_archive(database: WritableDatabase, file: UploadFile = File(...)) -> AddPhotosOut:
    _require_idle(database)
    limits = _limits_for(database["user_id"])
    if limits.max_total_bytes == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")

    staging = _tmp_dir(database, f"import-{db.new_id()}")
    archive_path = staging / "upload.zip"
    try:
        _save_upload(file, archive_path, limits.max_total_bytes)
        # проверка до распаковки: и битый архив, и zip-бомба должны отсеяться
        # раньше, чем на диск ляжет хоть один распакованный байт
        count, _ = inspect(archive_path, limits)
    except ArchiveError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        _check_room(database, count)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    def task(context: JobContext) -> str:
        try:
            context.set_message("Распаковка архива")
            extracted = extract_images(archive_path, staging / "files", limits)
            context.check_cancelled()

            context.set_message("Индексация")
            store = store_for(database)
            outcome = _index_files(store, extracted.files, context)
            skipped = len(outcome["skipped"]) + len(extracted.skipped)
            return f"Добавлено фото: {outcome['added']}, пропущено: {skipped}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            store = store_for(database)
            sync_stats(database["id"], store)

    job = job_queue.submit(
        kind="import_zip",
        user_id=database["user_id"],
        database_id=database["id"],
        function=task,
        total=count,
    )
    return AddPhotosOut(job_id=job["id"])


# --------------------------------------------------------------------------
# Удаление
# --------------------------------------------------------------------------

@router.delete("/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_photo(database: WritableDatabase, photo_id: str) -> None:
    store = store_for(database)
    if store.delete_photos([photo_id]) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")
    sync_stats(database["id"], store)


@router.post("/photos/delete")
def delete_photos(database: WritableDatabase, payload: DeletePhotosRequest) -> dict:
    store = store_for(database)
    removed = store.delete_photos(payload.photo_ids)
    sync_stats(database["id"], store)
    return {"deleted": removed}
