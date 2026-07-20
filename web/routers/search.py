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

from web.deps import CurrentUser, OwnedDatabase
from web.schemas import CaptionHitOut, SearchHitOut, SearchResultOut, SearchTextRequest
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


def _hits(database: dict, hits) -> list[SearchHitOut]:
    base = f"/api/databases/{database['id']}/photos"
    return [
        SearchHitOut(
            photo_id=hit.photo_id,
            score=round(hit.score, 4),
            thumb_url=f"{base}/{hit.photo_id}/thumb",
            file_url=f"{base}/{hit.photo_id}/file",
            caption=hit.caption,
        )
        for hit in hits
    ]


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
    )
    return SearchResultOut(
        used_query=used_query,
        results=_hits(database, hits),
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
    hits, captions = store.search_image(image, top_k=max(1, min(top_k, 50)))
    return SearchResultOut(
        results=_hits(database, hits),
        captions=[
            CaptionHitOut(photo_id=c.photo_id, score=round(c.score, 4), caption=c.caption)
            for c in captions
        ],
    )
