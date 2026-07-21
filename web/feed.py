"""
Персональная лента (web/routers/feed.py) — Pinterest-подобная главная страница.

Два источника, оба переиспользуют то, что человек уже сделал:
1. «Похожее» — core.store.IndexStore.search_similar по недавним лайкам,
   избранному и просмотрам (эмбеддинг не считается заново, вектор уже в индексе).
2. Повтор недавних текстовых запросов тем же used_query (без повторного перевода).

Каждый источник ограничен по top_k, иначе один недавний лайк забил бы всю ленту
похожими друг на друга снимками. Результат дедуплицируется по (база, фото) —
если снимок нашёлся через несколько сидов, остаётся с лучшим счётом.
"""

from __future__ import annotations

from web import db, services
from web.config import get_settings
from web.schemas import SearchHitOut
from web.stores import caption_encoder_for, store_for

PER_SEED_TOP_K = 6
PER_QUERY_TOP_K = 8
SEED_LIMIT = 15
QUERY_LIMIT = 5
FALLBACK_PER_DATABASE = 8


def build_feed(user: dict, limit: int = 60) -> list[SearchHitOut]:
    settings = get_settings()
    user_id = user["id"]
    databases = {d["id"]: d for d in db.list_databases(user_id)}

    best: dict[tuple[str, str], SearchHitOut] = {}

    def consider(database: dict, hit_outs: list[SearchHitOut]) -> None:
        for hit in hit_outs:
            key = (database["id"], hit.photo_id)
            current = best.get(key)
            if current is None or hit.score > current.score:
                best[key] = hit

    seeds_by_db: dict[str, list[str]] = {}
    for row in db.recent_interacted_photos(user_id, limit=SEED_LIMIT):
        seeds_by_db.setdefault(row["database_id"], []).append(row["photo_id"])

    for database_id, photo_ids in seeds_by_db.items():
        database = databases.get(database_id)
        if database is None:
            continue  # база с тех пор стала недоступна (вышел из чата и т.п.)
        store = store_for(database)
        for photo_id in photo_ids:
            hits = store.search_similar(
                photo_id, top_k=PER_SEED_TOP_K, min_score=settings.search_image_min_score,
            )
            consider(database, services.hits_out(database, hits, user_id))

    for row in db.recent_queries(user_id, limit=QUERY_LIMIT):
        database = databases.get(row["database_id"])
        if database is None:
            continue
        store = store_for(database)
        encoder = caption_encoder_for(store)
        # translate=False: used_query уже переведён на английский при первом
        # поиске (или это исходный текст, если он и так был на английском) —
        # переводить второй раз незачем и рискованно (перевод не идемпотентен).
        _, hits = store.search_text(
            row["used_query"], top_k=PER_QUERY_TOP_K, translate=False,
            caption_encoder=encoder, min_score=settings.search_text_min_score,
        )
        consider(database, services.hits_out(database, hits, user_id))

    if best:
        return sorted(best.values(), key=lambda hit: hit.score, reverse=True)[:limit]

    # Совсем нет истории (только что зарегистрировался) — лента не должна быть
    # пустым экраном: показываем недавно добавленные снимки из видимых баз.
    fallback: list[SearchHitOut] = []
    for database in databases.values():
        store = store_for(database)
        photos = store.list_photos(0, FALLBACK_PER_DATABASE)
        hits = [_as_hit(photo) for photo in photos]
        fallback.extend(services.hits_out(database, hits, user_id))
    return fallback[:limit]


class _FakeHit:
    """Оборачивает Photo в форму, которую ждёт services.hits_out (photo_id/score/caption)."""

    __slots__ = ("photo_id", "score", "caption")

    def __init__(self, photo_id: str, caption: str) -> None:
        self.photo_id = photo_id
        self.score = 0.0
        self.caption = caption


def _as_hit(photo) -> _FakeHit:
    return _FakeHit(photo.photo_id, photo.caption)
