"""
Зависимости FastAPI: текущий пользователь и проверка владения базой.

Проверка владельца вынесена в зависимость сознательно: если писать её вручную в каждой
ручке с {database_id}, рано или поздно в одной забудут — и чужая база станет доступна
по прямой ссылке (docs/WEB_PLAN.md, раздел 7).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Cookie, Depends, HTTPException, Request, status

from web import db
from web.config import Settings, get_settings
from web.security import SESSION_COOKIE, hash_token


def settings_dep() -> Settings:
    return get_settings()


def client_ip(request: Request) -> str:
    """
    IP клиента с учётом обратного прокси.

    Без этого в продакшене ограничение частоты входа превращается в отказ в
    обслуживании: за nginx все запросы приходят с одного адреса (адреса контейнера),
    поэтому пятая неудачная попытка входа любого пользователя заблокировала бы вход
    сразу всем.

    X-Real-IP берётся первым: nginx проставляет его сам и всегда перезаписывает,
    поэтому подделать его клиент не может. В X-Forwarded-For доверяем только
    последнему элементу — его дописывает прокси, всё, что левее, пришло от клиента
    и может быть выдумано.
    """
    if get_settings().trust_proxy:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def get_current_user(
    request: Request,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> dict[str, Any]:
    if not session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Требуется вход")

    token_hash = hash_token(session)
    row = db.get_session(token_hash)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Сессия не найдена")
    if db.is_expired(row["expires_at"]):
        db.delete_session(token_hash)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Сессия истекла")

    user = db.get_user(row["user_id"])
    if user is None:
        db.delete_session(token_hash)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Пользователь удалён")

    request.state.session_token_hash = token_hash
    return user


CurrentUser = Annotated[dict, Depends(get_current_user)]


def get_owned_database(database_id: str, user: CurrentUser) -> dict[str, Any]:
    """
    Возвращает базу пользователя. Чужая и несуществующая база дают одинаковый 404:
    иначе по коду ответа можно перебирать существующие id.
    """
    database = db.get_database(database_id)
    if database is None or database["user_id"] != user["id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "База не найдена")
    return database


OwnedDatabase = Annotated[dict, Depends(get_owned_database)]
