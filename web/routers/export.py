"""
Выгрузка базы одним zip-архивом.

Фоновая задача здесь не нужна: без сжатия (фото и так JPEG) архив собирается на лету
из уже готовых файлов, поэтому отдаётся потоком сразу — ждать «подготовку экспорта»
пользователю не приходится.
"""

from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from web.archive import stream_zip
from web.deps import OwnedDatabase
from web.stores import store_cache

router = APIRouter(prefix="/api/databases/{database_id}", tags=["export"])


def _content_disposition(name: str) -> str:
    """
    Имя файла для скачивания. Базу могли назвать «Отпуск 2024», а в заголовке HTTP
    разрешён только ASCII, поэтому кириллица уходит в filename* по RFC 5987,
    а в filename остаётся безопасная замена для старых клиентов.
    """
    # проверка на ASCII обязательна: str.isalnum() истинен и для кириллицы, поэтому
    # без неё «Отпуск 2024» попадёт в заголовок как есть и уронит ответ на latin-1
    safe = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in " -_") else "_" for ch in name
    ).strip("_ ") or "database"
    return f"attachment; filename=\"{safe}.zip\"; filename*=UTF-8''{quote(name + '.zip')}"


@router.get("/export.zip")
def export_database(database: OwnedDatabase) -> StreamingResponse:
    store = store_cache.get(database["user_id"], database["id"])
    manifest = json.dumps(store.manifest(), ensure_ascii=False, indent=2).encode("utf-8")

    return StreamingResponse(
        stream_zip(store.iter_export_files(), manifest=manifest),
        media_type="application/zip",
        headers={
            "Content-Disposition": _content_disposition(database["name"]),
            # длину заранее не знаем — архив собирается на лету
            "Cache-Control": "no-store",
        },
    )
