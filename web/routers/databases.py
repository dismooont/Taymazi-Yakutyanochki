"""
CRUD пользовательских баз.

Загрузка фото, поиск и экспорт — этапы M2–M4; здесь только жизненный цикл самой базы
и её объём, который фронт показывает в списке («1 240 фото · 2,1 ГБ»).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from web import db
from web.config import get_settings
from web.deps import CurrentUser, OwnedDatabase
from web.schemas import CreateDatabaseRequest, DatabaseOut, QuotaOut, RenameDatabaseRequest
from web.stores import create_store, remove_store_files, store_cache, sync_stats

router = APIRouter(prefix="/api/databases", tags=["databases"])


@router.get("", response_model=list[DatabaseOut])
def list_databases(user: CurrentUser) -> list[DatabaseOut]:
    return [DatabaseOut.from_row(row) for row in db.list_databases(user["id"])]


@router.post("", response_model=DatabaseOut, status_code=status.HTTP_201_CREATED)
def create_database(payload: CreateDatabaseRequest, user: CurrentUser) -> DatabaseOut:
    settings = get_settings()

    if db.count_databases(user["id"]) >= settings.max_db_per_user:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Достигнут лимит баз ({settings.max_db_per_user}). Удалите ненужную",
        )
    if db.user_total_bytes(user["id"]) >= settings.max_bytes_per_user:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")

    row = db.create_database(user["id"], payload.name.strip())
    try:
        store = create_store(user["id"], row["id"])
    except Exception:
        # запись в БД без папки на диске хуже, чем отсутствие обеих
        db.delete_database(row["id"])
        raise
    return DatabaseOut.from_row(sync_stats(row["id"], store))


@router.get("/{database_id}", response_model=DatabaseOut)
def get_database(database: OwnedDatabase) -> DatabaseOut:
    return DatabaseOut.from_row(database)


@router.get("/{database_id}/stats", response_model=DatabaseOut)
def get_stats(database: OwnedDatabase) -> DatabaseOut:
    """
    Пересчитывает объём по реальному состоянию базы. Список баз отдаёт счётчики из БД
    (быстро), а эта ручка — источник правды, если они разошлись после сбоя.
    """
    store = store_cache.get(database["user_id"], database["id"])
    return DatabaseOut.from_row(sync_stats(database["id"], store))


@router.patch("/{database_id}", response_model=DatabaseOut)
def rename_database(payload: RenameDatabaseRequest, database: OwnedDatabase) -> DatabaseOut:
    db.rename_database(database["id"], payload.name.strip())
    return DatabaseOut.from_row(db.get_database(database["id"]))


@router.delete("/{database_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_database(database: OwnedDatabase) -> None:
    db.delete_database(database["id"])
    remove_store_files(database["user_id"], database["id"])


quota_router = APIRouter(prefix="/api/quota", tags=["databases"])


@quota_router.get("", response_model=QuotaOut)
def get_quota(user: CurrentUser) -> QuotaOut:
    settings = get_settings()
    return QuotaOut(
        databases_used=db.count_databases(user["id"]),
        databases_limit=settings.max_db_per_user,
        bytes_used=db.user_total_bytes(user["id"]),
        bytes_limit=settings.max_bytes_per_user,
        photos_per_database_limit=settings.max_photos_per_db,
    )
