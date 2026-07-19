"""
Слой хранения: SQLite (пользователи, сессии, базы, задачи).

Почему SQLite, а не JSON-файлы: нужны транзакции при обновлении счётчиков объёма
и уникальность логина. Почему не Postgres: приложение однопроцессное (docs/WEB_PLAN.md,
раздел 2), нагрузка — десятки пользователей, лишний контейнер того не стоит.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from web.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    login         TEXT UNIQUE,                 -- всегда в нижнем регистре
    display_name  TEXT NOT NULL,
    password_hash TEXT,                        -- NULL, если вход только через Telegram
    email         TEXT,
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT
);

CREATE TABLE IF NOT EXISTS identities (
    provider         TEXT NOT NULL,            -- 'telegram'
    provider_user_id TEXT NOT NULL,
    user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    linked_at        TEXT NOT NULL,
    PRIMARY KEY (provider, provider_user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash   TEXT PRIMARY KEY,             -- sha256 токена, не сам токен
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    last_seen_at TEXT,
    user_agent   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS databases (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    photos_count INTEGER NOT NULL DEFAULT 0,
    photos_bytes INTEGER NOT NULL DEFAULT 0,
    index_bytes  INTEGER NOT NULL DEFAULT 0,
    has_captions INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready',   -- ready | indexing | error
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_databases_user ON databases(user_id);

CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    database_id    TEXT,
    kind           TEXT NOT NULL,              -- import_zip | add_photos | export_zip
    status         TEXT NOT NULL,              -- queued | running | done | error
    progress_done  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    message        TEXT,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_database ON jobs(database_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def expires_in(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")


def is_expired(iso_timestamp: str) -> bool:
    try:
        return datetime.fromisoformat(iso_timestamp) <= datetime.now(timezone.utc)
    except ValueError:
        return True  # неразбираемая дата — считаем сессию протухшей


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """
    Соединение на операцию. WAL позволяет читать во время записи — иначе поиск в вебе
    подвисал бы на время работы фоновой индексации.
    """
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Пользователи
# --------------------------------------------------------------------------

def create_user(login: str | None, display_name: str, password_hash: str | None,
                email: str | None = None) -> dict[str, Any]:
    user_id = new_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO users (id, login, display_name, password_hash, email, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, login, display_name, password_hash, email, now()),
        )
    return get_user(user_id)


def get_user(user_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_login(login: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE login = ?", (login.lower(),)).fetchone()
    return dict(row) if row else None


def set_password_hash(user_id: str, password_hash: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))


def touch_user(user_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now(), user_id))


# --------------------------------------------------------------------------
# Внешние способы входа (Telegram) — модель заложена сразу, ручки появятся в M1b
# --------------------------------------------------------------------------

def link_identity(provider: str, provider_user_id: str, user_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO identities (provider, provider_user_id, user_id, linked_at)"
            " VALUES (?, ?, ?, ?)",
            (provider, str(provider_user_id), user_id, now()),
        )


def get_user_by_identity(provider: str, provider_user_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM users u JOIN identities i ON i.user_id = u.id"
            " WHERE i.provider = ? AND i.provider_user_id = ?",
            (provider, str(provider_user_id)),
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------
# Сессии
# --------------------------------------------------------------------------

def create_session(user_id: str, token_hash: str, ttl_days: int, user_agent: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen_at,"
            " user_agent) VALUES (?, ?, ?, ?, ?, ?)",
            (token_hash, user_id, now(), expires_in(ttl_days), now(), user_agent[:200]),
        )


def get_session(token_hash: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
    return dict(row) if row else None


def delete_session(token_hash: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def delete_user_sessions(user_id: str, keep_token_hash: str | None = None) -> int:
    with connect() as conn:
        if keep_token_hash:
            cur = conn.execute(
                "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
                (user_id, keep_token_hash),
            )
        else:
            cur = conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return cur.rowcount


def purge_expired_sessions() -> int:
    with connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now(),))
        return cur.rowcount


# --------------------------------------------------------------------------
# Базы
# --------------------------------------------------------------------------

def create_database(user_id: str, name: str) -> dict[str, Any]:
    database_id = new_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO databases (id, user_id, name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (database_id, user_id, name, now(), now()),
        )
    return get_database(database_id)


def get_database(database_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM databases WHERE id = ?", (database_id,)).fetchone()
    return dict(row) if row else None


def list_databases(user_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM databases WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def count_databases(user_id: str) -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM databases WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def user_total_bytes(user_id: str) -> int:
    """Сколько всего занимают базы пользователя — для проверки квоты."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(photos_bytes + index_bytes), 0) FROM databases WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row[0])


def rename_database(database_id: str, name: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE databases SET name = ?, updated_at = ? WHERE id = ?",
            (name, now(), database_id),
        )


def update_database_stats(database_id: str, *, photos_count: int, photos_bytes: int,
                          index_bytes: int, has_captions: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE databases SET photos_count = ?, photos_bytes = ?, index_bytes = ?,"
            " has_captions = ?, updated_at = ? WHERE id = ?",
            (photos_count, photos_bytes, index_bytes, int(has_captions), now(), database_id),
        )


def set_database_status(database_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE databases SET status = ?, updated_at = ? WHERE id = ?",
            (status, now(), database_id),
        )


def delete_database(database_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM databases WHERE id = ?", (database_id,))
