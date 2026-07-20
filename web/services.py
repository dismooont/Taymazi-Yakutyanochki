"""
Общая логика работы с базами — одна на все входы.

До появления этого модуля бот вызывал IndexStore напрямую, а веб шёл через свои
обработчики. Получалось два пути к одному действию: у ботовского не было ни квот,
ни очереди, ни защиты от одновременных операций. Теперь и HTTP-ручки сайта, и ручки
бота (web/routers/bot.py) вызывают отсюда, поэтому правило, добавленное в одном
месте, действует везде.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from core.store import IndexStore
from web import db
from web.archive import ArchiveLimits
from web.config import get_settings
from web.jobs import JobContext, job_queue
from web.stores import create_store, store_for, sync_stats

# До этого числа файлов включительно работаем синхронно: ждать поллинга ради одного
# снимка — хуже, чем подождать секунду ответа.
SYNC_UPLOAD_LIMIT = 3


# --------------------------------------------------------------------------
# Проверки, общие для всех входов
# --------------------------------------------------------------------------

def limits_for(user_id: str) -> ArchiveLimits:
    """Лимит на архив — это остаток дисковой квоты пользователя, а не отдельное число."""
    settings = get_settings()
    remaining = max(0, settings.max_bytes_per_user - db.user_total_bytes(user_id))
    return ArchiveLimits(max_total_bytes=remaining)


def check_room(database: dict, incoming: int) -> None:
    settings = get_settings()
    if database["photos_count"] + incoming > settings.max_photos_per_db:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"В базе не более {settings.max_photos_per_db} фото "
            f"(сейчас {database['photos_count']})",
        )
    if db.user_total_bytes(database["user_id"]) >= settings.max_bytes_per_user:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")


def require_idle(database: dict) -> None:
    """
    Две одновременные операции над одной базой не имеют смысла: они всё равно
    выстроятся в очередь, но пользователь увидит два конкурирующих прогресс-бара.
    """
    if db.has_active_job(database["id"]):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "База уже обрабатывается, дождитесь окончания"
        )


def require_writable(database: dict) -> None:
    if database.get("read_only"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "База доступна только для чтения")


def tmp_dir(database: dict, name: str) -> Path:
    root = get_settings().database_dir(database["user_id"], database["id"]) / "tmp" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# Индексация
# --------------------------------------------------------------------------

def index_files(store: IndexStore, files: list[Path], context: JobContext | None,
                names: dict[str, str] | None = None) -> dict:
    """
    names — соответствие «имя во временной папке -> имя, которое дал файлу пользователь».
    Во временной папке имена снабжены порядковым префиксом (чтобы два файла с одинаковым
    именем не затёрли друг друга), но показывать этот префикс в списке пропущенных нельзя:
    пользователь не найдёт у себя «0001_photo.jpg».
    """
    result = store.add_photos(files, on_progress=(context.progress if context else None))
    skipped = [((names or {}).get(name, name), reason) for name, reason in result.skipped]
    return {
        "added": result.added_count,
        "skipped": skipped,
        # id первого добавленного нужен боту, чтобы сразу показать похожие
        "photo_id": result.added[0].photo_id if result.added else None,
    }


def add_photo_paths(database: dict, files: list[Path], staging: Path,
                    names: dict[str, str] | None = None) -> dict:
    """
    Добавляет уже лежащие на диске файлы: проверки, затем либо сразу, либо очередью.
    Возвращает {"job_id": ..., "added": ..., "skipped": [...]}.

    staging удаляется в любом случае — файлы к этому моменту скопированы внутрь базы.
    """
    if not files:
        shutil.rmtree(staging, ignore_errors=True)
        return {"job_id": None, "added": 0, "skipped": [], "photo_id": None}

    if len(files) <= SYNC_UPLOAD_LIMIT:
        try:
            store = store_for(database)
            outcome = index_files(store, files, None, names)
            sync_stats(database["id"], store)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"job_id": None, **outcome}

    def task(context: JobContext) -> str:
        try:
            store = store_for(database)
            outcome = index_files(store, files, context, names)
            return f"Добавлено фото: {outcome['added']}, пропущено: {len(outcome['skipped'])}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            sync_stats(database["id"], store_for(database))

    job = job_queue.submit(
        kind="add_photos",
        user_id=database["user_id"],
        database_id=database["id"],
        function=task,
        total=len(files),
    )
    return {"job_id": job["id"], "added": 0, "skipped": [], "photo_id": None}


# --------------------------------------------------------------------------
# Чаты Telegram
# --------------------------------------------------------------------------

def account_for_telegram(telegram_id: str, display_name: str) -> dict[str, Any]:
    """
    Аккаунт, соответствующий пользователю Telegram. Тот же самый, в который человек
    попадёт, войдя на сайт через Telegram, — благодаря общей таблице identities.
    """
    user = db.get_user_by_identity("telegram", telegram_id)
    if user is None:
        user = db.create_user(
            login=None, display_name=display_name or f"tg{telegram_id}", password_hash=None
        )
        db.link_identity("telegram", telegram_id, user["id"])
    return user


def ensure_chat_database(chat_id: str, title: str, owner: dict[str, Any]) -> dict[str, Any]:
    """Возвращает базу чата, создавая её при первом обращении."""
    database = db.get_database_by_chat(chat_id)
    if database is not None:
        return database

    database = db.create_database(owner["id"], title, kind="chat", telegram_chat_id=str(chat_id))
    try:
        create_store(owner["id"], database["id"])
    except Exception:
        db.delete_database(database["id"])
        raise
    return sync_stats(database["id"], store_for(database))


def refresh_stats(database: dict) -> dict:
    return sync_stats(database["id"], store_for(database))
