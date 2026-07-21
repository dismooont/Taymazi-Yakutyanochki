"""
Ручки для Telegram-бота.

Бот — отдельный процесс без модели и без доступа к файлам индекса: он умеет только
разговаривать по HTTP. Всё, что он делает с базой, проходит через тот же сервисный
слой, что и запросы с сайта (web/services.py), поэтому квоты, очередь задач и
запрет на изменение демо-базы действуют одинаково.

Аутентификация — служебный токен в заголовке. Это доверенный внутренний клиент:
он сообщает, от чьего имени действует, и API ему верит. Поэтому порт API наружу
не публикуется, а токен должен быть длинным и случайным.
"""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from web import db, services
from web.config import get_settings
from web.routers.photos import resolve_photo_file
from web.schemas import BotChatOut, BotSearchRequest, BotSearchResultOut, SearchHitOut
from web.stores import store_for

router = APIRouter(prefix="/api/bot", tags=["bot"])

MAX_PHOTO_BYTES = 30 * 1024 ** 2


def require_service_token(
    x_service_token: Annotated[str | None, Header()] = None,
) -> None:
    """
    Проверка служебного токена. Сравнение с постоянным временем: обычное ==
    позволяет подбирать токен посимвольно по времени ответа.
    """
    expected = get_settings().service_token
    if not expected:
        # токен не настроен — ручек бота как будто нет вовсе
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Не найдено")
    if not x_service_token or not secrets.compare_digest(x_service_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный служебный токен")


ServiceAuth = Annotated[None, Depends(require_service_token)]


def _chat_database(chat_id: str) -> dict[str, Any]:
    database = db.get_database_by_chat(str(chat_id))
    if database is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "В этом чате ещё нет базы")
    return database


# --------------------------------------------------------------------------
# Чат и его база
# --------------------------------------------------------------------------

@router.post("/chats/{chat_id}/start", response_model=BotChatOut)
def start_chat(
    chat_id: str,
    _: ServiceAuth,
    telegram_user_id: str = Form(...),
    display_name: str = Form(""),
    title: str = Form(""),
) -> BotChatOut:
    """Создаёт базу чата при первом /start и запоминает отправителя как участника."""
    user = services.account_for_telegram(telegram_user_id, display_name)
    existed = db.get_database_by_chat(str(chat_id)) is not None
    database = services.ensure_chat_database(str(chat_id), title or f"Чат {chat_id}", user)
    db.remember_chat_member(chat_id, user["id"])
    return BotChatOut.from_row(database, created=not existed)


@router.get("/chats/{chat_id}", response_model=BotChatOut)
def chat_info(chat_id: str, _: ServiceAuth) -> BotChatOut:
    database = _chat_database(chat_id)
    return BotChatOut.from_row(services.refresh_stats(database))


@router.post("/chats/{chat_id}/members", status_code=status.HTTP_204_NO_CONTENT)
def remember_member(
    chat_id: str, _: ServiceAuth,
    telegram_user_id: str = Form(...), display_name: str = Form(""),
) -> None:
    """
    Отмечает человека участником чата. По этим записям сайт решает, показывать ли
    ему базу чата, поэтому бот шлёт их при каждом сообщении.
    """
    user = services.account_for_telegram(telegram_user_id, display_name)
    db.remember_chat_member(chat_id, user["id"])


# --------------------------------------------------------------------------
# Фотографии и поиск
# --------------------------------------------------------------------------

@router.post("/chats/{chat_id}/photos", response_model=BotChatOut)
def add_photo(
    chat_id: str, _: ServiceAuth,
    file: UploadFile = File(...),
    telegram_user_id: str = Form(""),
    display_name: str = Form(""),
) -> BotChatOut:
    database = _chat_database(chat_id)
    services.require_writable(database)
    services.require_idle(database)
    services.check_room(database, 1)

    if telegram_user_id:
        user = services.account_for_telegram(telegram_user_id, display_name)
        db.remember_chat_member(chat_id, user["id"])

    staging = services.tmp_dir(database, f"tg-{db.new_id()}")
    name = Path(file.filename or "photo.jpg").name
    target = staging / name
    try:
        data = file.file.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Слишком большое фото")
        target.write_bytes(data)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    outcome = services.add_photo_paths(database, [target], staging, {name: name})
    fresh = db.get_database(database["id"])
    return BotChatOut.from_row(
        fresh, added=outcome["added"], photo_id=outcome["photo_id"], skipped=outcome["skipped"]
    )


def _bot_hit(chat_id: str, photo_id: str, score: float, *, ai_generated: bool = False) -> SearchHitOut:
    url = f"/api/bot/chats/{chat_id}/photos/{photo_id}/file"
    return SearchHitOut(
        photo_id=photo_id, score=round(score, 4), thumb_url=url, file_url=url,
        ai_generated=ai_generated,
    )


@router.post("/chats/{chat_id}/search", response_model=BotSearchResultOut)
def search_chat(chat_id: str, payload: BotSearchRequest, _: ServiceAuth) -> BotSearchResultOut:
    database = _chat_database(chat_id)
    store = store_for(database)
    used_query, hits = store.search_text(
        payload.query.strip(), top_k=payload.top_k, translate=payload.translate,
        min_score=get_settings().search_text_min_score,
    )
    results = [_bot_hit(chat_id, hit.photo_id, hit.score) for hit in hits]

    # Квота привязана к владельцу базы чата (тому, кто запустил /start) — у
    # чата нет отдельного «текущего пользователя», а квота на человека, а не
    # на чат, не даёт исчерпать общий ключ одной активной группой.
    if not results:
        generated = services.generate_fallback_photo(database, database["user_id"], used_query or payload.query)
        if generated:
            results = [_bot_hit(chat_id, generated["photo_id"], 1.0, ai_generated=True)]

    return BotSearchResultOut(used_query=used_query, results=results)


@router.get("/chats/{chat_id}/photos/{photo_id}/similar", response_model=BotSearchResultOut)
def similar_photos(chat_id: str, photo_id: str, _: ServiceAuth, top_k: int = 5) -> BotSearchResultOut:
    """
    Похожие на снимок, уже лежащий в базе. Именно так работает сценарий
    «прислал фото — покажи похожие»: эмбеддинг не считается заново, вектор
    берётся из индекса.
    """
    database = _chat_database(chat_id)
    store = store_for(database)
    hits = store.search_similar(
        photo_id, top_k=max(1, min(top_k, 20)),
        min_score=get_settings().search_image_min_score,
    )
    return BotSearchResultOut(
        used_query="",
        results=[_bot_hit(chat_id, hit.photo_id, hit.score) for hit in hits],
    )


@router.get("/chats/{chat_id}/photos/{photo_id}/file")
def photo_file(chat_id: str, photo_id: str, _: ServiceAuth) -> FileResponse:
    """Бот забирает найденный снимок, чтобы отправить его в чат."""
    database = _chat_database(chat_id)
    return FileResponse(resolve_photo_file(database, photo_id, thumb=False))


# --------------------------------------------------------------------------
# Демо-база
# --------------------------------------------------------------------------

def _demo_database() -> dict[str, Any]:
    """
    Общая база MS COCO. Доступна из любого чата и только на чтение, поэтому
    ни владения, ни участия здесь не проверяется — проверять нечего.
    """
    database = db.get_demo_database()
    if database is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Демо-база не подключена")
    return database


@router.get("/demo", response_model=BotChatOut)
def demo_info(_: ServiceAuth) -> BotChatOut:
    return BotChatOut.from_row(_demo_database())


@router.post("/demo/search", response_model=BotSearchResultOut)
def search_demo(payload: BotSearchRequest, _: ServiceAuth) -> BotSearchResultOut:
    database = _demo_database()
    store = store_for(database)
    used_query, hits = store.search_text(
        payload.query.strip(), top_k=payload.top_k, translate=payload.translate,
        min_score=get_settings().search_text_min_score,
    )
    return BotSearchResultOut(
        used_query=used_query,
        results=[
            SearchHitOut(
                photo_id=hit.photo_id,
                score=round(hit.score, 4),
                thumb_url=f"/api/bot/demo/photos/{hit.photo_id}/file",
                file_url=f"/api/bot/demo/photos/{hit.photo_id}/file",
            )
            for hit in hits
        ],
    )


@router.get("/demo/photos/{photo_id}/file")
def demo_photo_file(photo_id: str, _: ServiceAuth) -> FileResponse:
    return FileResponse(resolve_photo_file(_demo_database(), photo_id, thumb=False))
