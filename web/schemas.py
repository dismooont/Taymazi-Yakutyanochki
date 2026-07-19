"""Схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Авторизация
# --------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    login: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=200)
    display_name: str | None = Field(default=None, max_length=64)
    email: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    login: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


class PasswordChangeRequest(BaseModel):
    old_password: str = Field(default="", max_length=200)
    new_password: str = Field(min_length=1, max_length=200)


class UserOut(BaseModel):
    id: str
    login: str | None
    display_name: str
    has_password: bool
    has_telegram: bool


# --------------------------------------------------------------------------
# Базы
# --------------------------------------------------------------------------

class CreateDatabaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class RenameDatabaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class DatabaseOut(BaseModel):
    id: str
    name: str
    photos_count: int
    photos_bytes: int
    index_bytes: int
    total_bytes: int
    has_captions: bool
    status: str
    # kind и read_only нужны интерфейсу, чтобы не предлагать действий, которые
    # заведомо получат отказ: у демо-базы нет ни удаления, ни добавления снимков
    kind: str = "personal"
    read_only: bool = False
    preview: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: dict) -> "DatabaseOut":
        raw_preview = row.get("preview") or ""
        return cls(
            kind=row.get("kind") or "personal",
            read_only=bool(row.get("read_only")),
            preview=[pid for pid in raw_preview.split(",") if pid],
            id=row["id"],
            name=row["name"],
            photos_count=row["photos_count"],
            photos_bytes=row["photos_bytes"],
            index_bytes=row["index_bytes"],
            total_bytes=row["photos_bytes"] + row["index_bytes"],
            has_captions=bool(row["has_captions"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PhotoOut(BaseModel):
    photo_id: str
    bytes: int
    added_at: str


class PhotoPageOut(BaseModel):
    total: int
    offset: int
    items: list[PhotoOut]


class AddPhotosOut(BaseModel):
    """
    Ответ на добавление фото. job_id заполнен, если работа ушла в очередь;
    added/skipped — если файлов было мало и они обработаны сразу.
    """

    job_id: str | None = None
    added: int = 0
    skipped: list[tuple[str, str]] = Field(default_factory=list)


class DeletePhotosRequest(BaseModel):
    photo_ids: list[str] = Field(min_length=1, max_length=1000)


class SearchTextRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=12, ge=1, le=50)
    translate: bool = True


class SearchHitOut(BaseModel):
    photo_id: str
    score: float
    thumb_url: str
    file_url: str


class CaptionHitOut(BaseModel):
    photo_id: str
    score: float
    caption: str


class SearchResultOut(BaseModel):
    """
    used_query — то, что реально ушло в CLIP: для русского запроса это его перевод.
    Фронт показывает это пользователю, иначе непонятно, почему «рыжий кот» нашёл
    именно эти снимки.
    """

    used_query: str | None = None
    results: list[SearchHitOut] = Field(default_factory=list)
    captions: list[CaptionHitOut] = Field(default_factory=list)


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    database_id: str | None
    progress_done: int
    progress_total: int
    queue_position: int
    message: str | None
    created_at: str
    finished_at: str | None

    @classmethod
    def from_row(cls, row: dict, queue_position: int = 0) -> "JobOut":
        return cls(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            database_id=row["database_id"],
            progress_done=row["progress_done"],
            progress_total=row["progress_total"],
            queue_position=queue_position,
            message=row["message"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )


class QuotaOut(BaseModel):
    """Лимиты и текущее потребление — фронт показывает их рядом с объёмом базы."""

    databases_used: int
    databases_limit: int
    bytes_used: int
    bytes_limit: int
    photos_per_database_limit: int
