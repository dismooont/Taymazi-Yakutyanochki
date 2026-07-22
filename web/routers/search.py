"""
Поиск по базе: по текстовому описанию и по картинке-образцу.

Поиск — единственная часть API, которая тратит CPU на каждый запрос без очереди
(один прогон энкодера, десятки миллисекунд), поэтому здесь стоит отдельное ограничение
частоты: приложение однопроцессное, и десяток параллельных «поисков» способен подвесить
интерфейс всем остальным.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from web import db, services
from web.config import get_settings
from web.deps import CurrentUser, OwnedDatabase
from web.schemas import CaptionHitOut, GenerateRequest, SearchResultOut, SearchTextRequest
from web.security import RateLimiter
from web.stores import caption_encoder_for, store_for

router = APIRouter(prefix="/api/databases/{database_id}/search", tags=["search"])

search_limiter = RateLimiter(limit=30, window_seconds=60)
MAX_QUERY_IMAGE_BYTES = 20 * 1024 ** 2


def _check_rate(user_id: str) -> None:
    if not search_limiter.check(f"search:{user_id}"):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Слишком много запросов, подождите минуту"
        )


@router.post("/text", response_model=SearchResultOut)
def search_by_text(
    payload: SearchTextRequest, database: OwnedDatabase, user: CurrentUser
) -> SearchResultOut:
    _check_rate(user["id"])
    store = store_for(database)
    encoder = caption_encoder_for(store)
    used_query, hits = store.search_text(
        payload.query.strip(),
        top_k=payload.top_k,
        translate=payload.translate,
        caption_encoder=encoder,
        min_score=get_settings().search_text_min_score,
    )
    # Косинус CLIP не отличает «нашлось по делу» от «нашлось похожее по одному
    # слову из нескольких» (см. services.filter_by_caption) — там, где уже есть
    # подпись BLIP, дополнительно проверяем, что она реально упоминает предметы
    # запроса. Снимки без подписи фильтр пропускает как есть.
    hits = services.filter_by_caption(hits, used_query or payload.query)
    results = services.hits_out(database, hits, user["id"])

    # Запрос запоминается для ленты (web/feed.py) независимо от того, нашлось
    # что-то или нет: used_query — то же самое подхватывает генерация ниже.
    db.log_search(user["id"], database["id"], payload.query.strip(), used_query or payload.query)

    # Поиск ничего не нашёл выше порога — пробуем сгенерировать снимок по тому
    # же тексту, что реально ушёл в модель (used_query — уже переведённый на
    # английский, если запрос был русским). Тихо остаётся пустой выдачей, если
    # генерация выключена, база read-only или дневной лимит исчерпан.
    if not results:
        generated = services.generate_fallback_photo(database, user["id"], used_query or payload.query)
        if generated:
            results = [services.generated_hit_out(database, generated["photo_id"])]

    return SearchResultOut(
        used_query=used_query,
        results=results,
        # Оценки слияния и обычного поиска — разные величины: у первого это
        # взвешенная сумма нормированных отклонений, у второго косинус. Показывать
        # их одинаково значило бы вводить человека в заблуждение.
        fused=encoder is not None,
    )


@router.post("/image", response_model=SearchResultOut)
def search_by_image(
    database: OwnedDatabase, user: CurrentUser, file: UploadFile = File(...), top_k: int = 12
) -> SearchResultOut:
    _check_rate(user["id"])

    raw = file.file.read(MAX_QUERY_IMAGE_BYTES + 1)
    if len(raw) > MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Слишком большое изображение-запрос"
        )
    try:
        # картинка-образец никуда не сохраняется: она нужна только на один прогон
        # энкодера, поэтому держим её в памяти и не засоряем диск
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Файл не является изображением"
        ) from e

    store = store_for(database)
    hits, captions = store.search_image(
        image, top_k=max(1, min(top_k, 50)),
        min_score=get_settings().search_image_min_score,
    )
    # То же ограничение, что и у текста, только с другой стороны: у богатой
    # предметной сцены (лампа на столе среди декора) косинус CLIP цепляется за
    # общую атмосферу «уютный интерьер», а не за лампу — в базе таких сцен
    # много, ламп почти нет. Подписываем сам запрос BLIP один раз и фильтруем
    # по ГЛАВНОМУ предмету (services.filter_by_image_caption) — не по всей
    # сцене целиком, иначе требовать совпадения стола и цветов рядом с лампой
    # означало бы искать композицию, а не предмет.
    query_caption = services.caption_query_image(image)
    hits = services.filter_by_image_caption(hits, query_caption)
    results = services.hits_out(database, hits, user["id"])

    # Пусто — пробуем сгенерировать по тому же описанию, что дала подпись
    # картинки-запроса: это и есть лучшее текстовое приближение того, что
    # искал человек. Как и в текстовом поиске, тихо остаётся пустой выдачей,
    # если генерация выключена, база read-only или лимит на сегодня исчерпан.
    if not results and query_caption:
        generated = services.generate_fallback_photo(database, user["id"], query_caption)
        if generated:
            results = [services.generated_hit_out(database, generated["photo_id"])]

    return SearchResultOut(
        used_query=query_caption,
        results=results,
        captions=[
            CaptionHitOut(photo_id=c.photo_id, score=round(c.score, 4), caption=c.caption)
            for c in captions
        ],
    )


@router.post("/generate", response_model=SearchResultOut)
def generate_photo(database: OwnedDatabase, user: CurrentUser, payload: GenerateRequest) -> SearchResultOut:
    """
    Генерация по явной команде человека, а не только когда поиск пуст.

    Порог косинуса не отличает «нашлось по делу» от «нашлось похожее по
    одному слову, но не то» — на запросе «лошадь на медведе» находятся
    медведи с оценкой ВЫШЕ, чем у корректного запроса «собака»: CLIP не
    проверяет отношения между объектами, только заметные существительные.
    Автоматика здесь бессильна, поэтому решение — за пользователем.
    """
    _check_rate(user["id"])
    settings = get_settings()
    if not settings.photo_generation_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Генерация фото не настроена на сервере")
    if database.get("read_only"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "В демо-базу генерировать нельзя")
    if db.count_generations_today(user["id"]) >= settings.yandex_generations_per_user_day:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Дневной лимит генераций исчерпан")

    query = payload.query.strip()
    generated = services.generate_fallback_photo(database, user["id"], query)
    if not generated:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "YandexART не смог сгенерировать снимок, попробуйте ещё раз"
        )
    return SearchResultOut(
        used_query=query, results=[services.generated_hit_out(database, generated["photo_id"])]
    )


@router.get("/similar/{photo_id}", response_model=SearchResultOut)
def search_similar(database: OwnedDatabase, user: CurrentUser, photo_id: str, top_k: int = 12) -> SearchResultOut:
    """
    Похожие на снимок, уже лежащий в базе — для страницы фото и для ленты.
    Эмбеддинг не считается заново (core.store.IndexStore.search_similar
    достаёт готовый вектор из индекса), поэтому дешевле обычного поиска.
    """
    _check_rate(user["id"])
    store = store_for(database)
    hits = store.search_similar(
        photo_id, top_k=max(1, min(top_k, 50)), min_score=get_settings().search_image_min_score,
    )
    return SearchResultOut(results=services.hits_out(database, hits, user["id"]))
