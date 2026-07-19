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
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: dict) -> "DatabaseOut":
        return cls(
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


class QuotaOut(BaseModel):
    """Лимиты и текущее потребление — фронт показывает их рядом с объёмом базы."""

    databases_used: int
    databases_limit: int
    bytes_used: int
    bytes_limit: int
    photos_per_database_limit: int
