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
    captions_count: int = 0  # сколько снимков уже размечено — для индикатора прогресса
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
            captions_count=row["captions_count"] if "captions_count" in row.keys() else 0,
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PhotoOut(BaseModel):
    photo_id: str
    bytes: int
    added_at: str
    caption: str = ""  # подпись снимка (ручная или сгенерированная), может быть пустой


class PhotoPageOut(BaseModel):
    total: int
    offset: int
    items: list[PhotoOut]


class SetCaptionRequest(BaseModel):
    # Пустая строка — снять подпись. Ограничение длины: подпись описывает один
    # снимок, а не текст поста; заодно это защита от раздувания меты.
    caption: str = Field(default="", max_length=500)


class CaptionOut(BaseModel):
    photo_id: str
    caption: str
    # попала ли подпись в поисковый индекс. False — сохранена только как текст
    # (нет sentence-transformers), видна и экспортируется, но на поиск не влияет.
    indexed: bool = False


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
    caption: str = ""  # подпись снимка, если она уже сгенерирована


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
    # Отработало ли слияние с поиском по подписям. Нужно фронту, чтобы не
    # показывать оценку слияния как косинус: это разные величины.
    fused: bool = False


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


class BotChatOut(BaseModel):
    """Состояние базы чата в том виде, в каком её показывает бот."""

    database_id: str
    name: str
    photos_count: int
    total_bytes: int
    created: bool = False
    added: int = 0
    # id только что добавленного снимка: по нему бот сразу просит похожие
    photo_id: str | None = None
    skipped: list[tuple[str, str]] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict, *, created: bool = False, added: int = 0,
                 photo_id: str | None = None, skipped: list | None = None) -> "BotChatOut":
        return cls(
            database_id=row["id"],
            name=row["name"],
            photos_count=row["photos_count"],
            total_bytes=row["photos_bytes"] + row["index_bytes"],
            created=created,
            added=added,
            photo_id=photo_id,
            skipped=skipped or [],
        )


class BotSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    translate: bool = True


class BotSearchResultOut(BaseModel):
    used_query: str
    results: list[SearchHitOut] = Field(default_factory=list)


class QuotaOut(BaseModel):
    """Лимиты и текущее потребление — фронт показывает их рядом с объёмом базы."""

    databases_used: int
    databases_limit: int
    bytes_used: int
    bytes_limit: int
    photos_per_database_limit: int
