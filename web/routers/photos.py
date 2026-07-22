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
from web.deps import CurrentUser, OwnedDatabase, WritableDatabase
from web import services
from web.jobs import JobContext, job_queue
from web.schemas import (
    AddPhotosOut,
    CaptionOut,
    DeletePhotosRequest,
    PhotoOut,
    PhotoPageOut,
    SetCaptionRequest,
)
from web.stores import database_root, set_manual_caption, store_for, sync_stats

router = APIRouter(prefix="/api/databases/{database_id}", tags=["photos"])

# До этого числа файлов включительно работаем синхронно: ждать поллинга ради одного
# снимка — хуже, чем подождать секунду ответа.
MAX_UPLOAD_FILES = 500
UPLOAD_CHUNK = 1 << 20


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------





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



# --------------------------------------------------------------------------
# Список фотографий
# --------------------------------------------------------------------------

@router.get("/photos", response_model=PhotoPageOut)
def list_photos(
    database: OwnedDatabase, user: CurrentUser, offset: int = 0, limit: int = 60
) -> PhotoPageOut:
    store = store_for(database)
    limit = max(1, min(limit, 200))
    photos = store.list_photos(offset=max(0, offset), limit=limit)
    photo_ids = [p.photo_id for p in photos]
    liked = db.liked_photo_ids(user["id"], database["id"], photo_ids)
    favorited = db.favorited_photo_ids(user["id"], database["id"], photo_ids)
    generated = db.ai_generated_photo_ids(database["id"], photo_ids)
    return PhotoPageOut(
        total=len(store),
        offset=offset,
        items=[
            PhotoOut(
                photo_id=p.photo_id, bytes=p.bytes, added_at=p.added_at, caption=p.caption,
                liked=p.photo_id in liked, favorited=p.photo_id in favorited,
                ai_generated=p.photo_id in generated,
            )
            for p in photos
        ],
    )


@router.get("/photos/{photo_id}/info", response_model=PhotoOut)
def get_photo_info(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> PhotoOut:
    """Один снимок — для отдельной страницы фото (не модалки), а не всей галереи."""
    store = store_for(database)
    photo = store.get_photo(photo_id)
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")
    liked = db.liked_photo_ids(user["id"], database["id"], [photo_id])
    favorited = db.favorited_photo_ids(user["id"], database["id"], [photo_id])
    generated = db.ai_generated_photo_ids(database["id"], [photo_id])
    return PhotoOut(
        photo_id=photo.photo_id, bytes=photo.bytes, added_at=photo.added_at, caption=photo.caption,
        liked=photo_id in liked, favorited=photo_id in favorited, ai_generated=photo_id in generated,
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
    services.require_idle(database)
    services.check_room(database, len(files))

    limits = services.limits_for(database["user_id"])
    staging = services.tmp_dir(database, f"upload-{db.new_id()}")
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

    # дальше — общая для сайта и бота логика: мало файлов обрабатываем сразу,
    # много отправляем в очередь
    outcome = services.add_photo_paths(database, saved, staging, names)
    return AddPhotosOut(**outcome)


# --------------------------------------------------------------------------
# Импорт архива
# --------------------------------------------------------------------------

@router.post("/import", response_model=AddPhotosOut, status_code=status.HTTP_202_ACCEPTED)
def import_archive(database: WritableDatabase, file: UploadFile = File(...)) -> AddPhotosOut:
    services.require_idle(database)
    limits = services.limits_for(database["user_id"])
    if limits.max_total_bytes == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")

    staging = services.tmp_dir(database, f"import-{db.new_id()}")
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
        services.check_room(database, count)
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
            outcome = services.index_files(store, extracted.files, context)
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
# Подпись вручную
# --------------------------------------------------------------------------

@router.put("/photos/{photo_id}/caption", response_model=CaptionOut)
def set_caption(
    database: WritableDatabase, photo_id: str, payload: SetCaptionRequest
) -> CaptionOut:
    """
    Задать или изменить подпись снимка. Пустая строка снимает подпись.

    WritableDatabase не пускает сюда демо-базу: она общая и только для чтения,
    подписи COCO трогать нельзя.
    """
    store = store_for(database)
    if store.get_photo(photo_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")

    indexed = set_manual_caption(store, photo_id, payload.caption)
    # покрытие подписями видно в шапке базы («размечено X из Y») — обновляем
    sync_stats(database["id"], store)
    return CaptionOut(photo_id=photo_id, caption=store.caption_of(photo_id), indexed=indexed)


# --------------------------------------------------------------------------
# Лайки и избранное
# --------------------------------------------------------------------------
# Отметка личная, не меняет саму базу — поэтому OwnedDatabase (даёт доступ и
# демо-базе, и базам чатов, где пользователь состоит), а не WritableDatabase.

def _existing_photo(database: dict, photo_id: str) -> None:
    if store_for(database).get_photo(photo_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")


@router.put("/photos/{photo_id}/like", status_code=status.HTTP_204_NO_CONTENT)
def like_photo(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> None:
    _existing_photo(database, photo_id)
    db.like_photo(user["id"], database["id"], photo_id)


@router.delete("/photos/{photo_id}/like", status_code=status.HTTP_204_NO_CONTENT)
def unlike_photo(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> None:
    db.unlike_photo(user["id"], database["id"], photo_id)


@router.put("/photos/{photo_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
def favorite_photo(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> None:
    _existing_photo(database, photo_id)
    db.favorite_photo(user["id"], database["id"], photo_id)


@router.delete("/photos/{photo_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
def unfavorite_photo(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> None:
    db.unfavorite_photo(user["id"], database["id"], photo_id)


# --------------------------------------------------------------------------
# Просмотр — сид для персональной ленты (web/feed.py), не для UI-состояния
# --------------------------------------------------------------------------

@router.post("/photos/{photo_id}/view", status_code=status.HTTP_204_NO_CONTENT)
def view_photo(database: OwnedDatabase, user: CurrentUser, photo_id: str) -> None:
    _existing_photo(database, photo_id)
    db.log_photo_view(user["id"], database["id"], photo_id)


# --------------------------------------------------------------------------
# Удаление
# --------------------------------------------------------------------------

@router.delete("/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_photo(database: WritableDatabase, photo_id: str) -> None:
    store = store_for(database)
    if store.delete_photos([photo_id]) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Фото не найдено")
    db.forget_photo_marks(database["id"], [photo_id])
    sync_stats(database["id"], store)


@router.post("/photos/delete")
def delete_photos(database: WritableDatabase, payload: DeletePhotosRequest) -> dict:
    store = store_for(database)
    removed = store.delete_photos(payload.photo_ids)
    db.forget_photo_marks(database["id"], payload.photo_ids)
    sync_stats(database["id"], store)
    return {"deleted": removed}
