"""
Общая логика работы с базами — одна на все входы.

До появления этого модуля бот вызывал IndexStore напрямую, а веб шёл через свои
обработчики. Получалось два пути к одному действию: у ботовского не было ни квот,
ни очереди, ни защиты от одновременных операций. Теперь и HTTP-ручки сайта, и ручки
бота (web/routers/bot.py) вызывают отсюда, поэтому правило, добавленное в одном
месте, действует везде.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from core import yandex_art
from core.store import IndexStore, SearchHit
from web import db
from web.archive import ArchiveLimits
from web.config import get_settings
from web.jobs import JobContext, job_queue
from web.schemas import SearchHitOut
from web.stores import create_store, store_for, sync_stats

# До этого числа файлов включительно работаем синхронно: ждать поллинга ради одного
# снимка — хуже, чем подождать секунду ответа.
SYNC_UPLOAD_LIMIT = 3


# --------------------------------------------------------------------------
# Проверки, общие для всех входов
# --------------------------------------------------------------------------

def limits_for(user_id: str) -> ArchiveLimits:
    """Лимит на архив — это остаток дисковой квоты пользователя, а не отдельное число."""
    settings = get_settings()
    remaining = max(0, settings.max_bytes_per_user - db.user_total_bytes(user_id))
    return ArchiveLimits(max_total_bytes=remaining)


def check_room(database: dict, incoming: int) -> None:
    settings = get_settings()
    if database["photos_count"] + incoming > settings.max_photos_per_db:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"В базе не более {settings.max_photos_per_db} фото "
            f"(сейчас {database['photos_count']})",
        )
    if db.user_total_bytes(database["user_id"]) >= settings.max_bytes_per_user:
        raise HTTPException(status.HTTP_409_CONFLICT, "Достигнут лимит объёма")


def require_idle(database: dict) -> None:
    """
    Две одновременные операции над одной базой не имеют смысла: они всё равно
    выстроятся в очередь, но пользователь увидит два конкурирующих прогресс-бара.
    """
    if db.has_active_job(database["id"]):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "База уже обрабатывается, дождитесь окончания"
        )


def require_writable(database: dict) -> None:
    if database.get("read_only"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "База доступна только для чтения")


def tmp_dir(database: dict, name: str) -> Path:
    root = get_settings().database_dir(database["user_id"], database["id"]) / "tmp" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# Индексация
# --------------------------------------------------------------------------

def index_files(store: IndexStore, files: list[Path], context: JobContext | None,
                names: dict[str, str] | None = None) -> dict:
    """
    names — соответствие «имя во временной папке -> имя, которое дал файлу пользователь».
    Во временной папке имена снабжены порядковым префиксом (чтобы два файла с одинаковым
    именем не затёрли друг друга), но показывать этот префикс в списке пропущенных нельзя:
    пользователь не найдёт у себя «0001_photo.jpg».
    """
    result = store.add_photos(files, on_progress=(context.progress if context else None))
    skipped = [((names or {}).get(name, name), reason) for name, reason in result.skipped]
    return {
        "added": result.added_count,
        "skipped": skipped,
        # id первого добавленного нужен боту, чтобы сразу показать похожие
        "photo_id": result.added[0].photo_id if result.added else None,
    }


def add_photo_paths(database: dict, files: list[Path], staging: Path,
                    names: dict[str, str] | None = None) -> dict:
    """
    Добавляет уже лежащие на диске файлы: проверки, затем либо сразу, либо очередью.
    Возвращает {"job_id": ..., "added": ..., "skipped": [...]}.

    staging удаляется в любом случае — файлы к этому моменту скопированы внутрь базы.
    """
    if not files:
        shutil.rmtree(staging, ignore_errors=True)
        return {"job_id": None, "added": 0, "skipped": [], "photo_id": None}

    if len(files) <= SYNC_UPLOAD_LIMIT:
        try:
            store = store_for(database)
            outcome = index_files(store, files, None, names)
            sync_stats(database["id"], store)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"job_id": None, **outcome}

    def task(context: JobContext) -> str:
        try:
            store = store_for(database)
            outcome = index_files(store, files, context, names)
            return f"Добавлено фото: {outcome['added']}, пропущено: {len(outcome['skipped'])}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            sync_stats(database["id"], store_for(database))

    job = job_queue.submit(
        kind="add_photos",
        user_id=database["user_id"],
        database_id=database["id"],
        function=task,
        total=len(files),
    )
    return {"job_id": job["id"], "added": 0, "skipped": [], "photo_id": None}


# --------------------------------------------------------------------------
# Чаты Telegram
# --------------------------------------------------------------------------

def account_for_telegram(telegram_id: str, display_name: str) -> dict[str, Any]:
    """
    Аккаунт, соответствующий пользователю Telegram. Тот же самый, в который человек
    попадёт, войдя на сайт через Telegram, — благодаря общей таблице identities.
    """
    user = db.get_user_by_identity("telegram", telegram_id)
    if user is None:
        user = db.create_user(
            login=None, display_name=display_name or f"tg{telegram_id}", password_hash=None
        )
        db.link_identity("telegram", telegram_id, user["id"])
    return user


def ensure_chat_database(chat_id: str, title: str, owner: dict[str, Any]) -> dict[str, Any]:
    """Возвращает базу чата, создавая её при первом обращении."""
    database = db.get_database_by_chat(chat_id)
    if database is not None:
        return database

    database = db.create_database(owner["id"], title, kind="chat", telegram_chat_id=str(chat_id))
    try:
        create_store(owner["id"], database["id"])
    except Exception:
        db.delete_database(database["id"])
        raise
    return sync_stats(database["id"], store_for(database))


def refresh_stats(database: dict) -> dict:
    return sync_stats(database["id"], store_for(database))


# --------------------------------------------------------------------------
# Построение SearchHitOut — общее для поиска (web/routers/search.py) и ленты
# (web/feed.py), чтобы отметки лайка/избранного/генерации не разъезжались
# --------------------------------------------------------------------------

def hits_out(database: dict, hits, user_id: str) -> list[SearchHitOut]:
    base = f"/api/databases/{database['id']}/photos"
    photo_ids = [hit.photo_id for hit in hits]
    liked = db.liked_photo_ids(user_id, database["id"], photo_ids)
    favorited = db.favorited_photo_ids(user_id, database["id"], photo_ids)
    generated = db.ai_generated_photo_ids(database["id"], photo_ids)
    return [
        SearchHitOut(
            photo_id=hit.photo_id,
            database_id=database["id"],
            score=round(hit.score, 4),
            thumb_url=f"{base}/{hit.photo_id}/thumb",
            file_url=f"{base}/{hit.photo_id}/file",
            caption=hit.caption,
            liked=hit.photo_id in liked,
            favorited=hit.photo_id in favorited,
            ai_generated=hit.photo_id in generated,
        )
        for hit in hits
    ]


# --------------------------------------------------------------------------
# Фильтр по подписи BLIP — отсекает «похоже только на один из предметов»
# --------------------------------------------------------------------------
# Косинус CLIP не отличает «нашлось по делу» от «нашлось похожее по одному
# слову из нескольких» — проверено эмпирически: у запроса «dolphin on
# horseback on bear» оценка ВЫШЕ, чем у корректного «dog» (0.30 против 0.27),
# и то же самое для z-score относительно фона базы. Отдельного порога здесь
# нет и не может быть. Реальная подпись снимка (BLIP) — другой, независимый
# сигнал: она либо действительно упоминает предметы запроса, либо нет.
#
# Снимки без подписи (разметка ещё не дошла или выключена) НЕ отсеиваются —
# отсутствие подписи не доказательство несовпадения, а просто нехватка данных.

_STOPWORDS = {
    "a", "an", "the", "on", "in", "at", "of", "with", "and", "or", "to", "is",
    "are", "his", "her", "its", "their", "this", "that", "photo", "image",
    "picture", "riding", "standing", "top", "some", "there",
    # Служебные «рамочные» слова BLIP: почти каждая десятая подпись в базе
    # начинается с «a close up of...», «a group of...», «a bunch of...» — без
    # этого списка _main_word брал бы «close»/«group»/«bunch» вместо реального
    # предмета (найдено эмпирически: фото медведя нашло только 1 совпадение
    # из полутора сотен медведей в базе, потому что «главным словом» подписи
    # запроса оказалось «close»).
    "close", "up", "group", "groups", "couple", "bunch", "pair", "pairs",
    "few", "several", "number", "lot", "shot", "view", "closeup",
}


def _words_in_order(text: str) -> list[str]:
    # Кириллица ловится тем же регулярным выражением намеренно: если перевод
    # разово не удался (сетевой сбой переводчика — сам он это не кэширует и
    # отдаёт исходный текст, core/translate.py), used_query останется русским.
    # Подписи BLIP всегда на английском, так что русские слова не совпадут ни
    # с одной из них — выдача уйдёт в генерацию вместо показа мусора, который
    # CLIP находит на сыром русском запросе (он на таком тексте не обучен).
    return [w for w in re.findall(r"[a-zа-яё]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]


def _content_words(text: str) -> set[str]:
    return set(_words_in_order(text))


def _main_word(caption: str) -> str | None:
    """
    Первое заметное слово подписи — обычно и есть главный предмет: BLIP
    почти всегда пишет «a <предмет> <где/с чем>» («a lamp on a table with
    flowers»), а не наоборот. Нужен для запроса-картинки (filter_by_image_caption):
    там подпись описывает всю сцену целиком, и требовать совпадения каждого
    слова («table», «flowers») значило бы искать композицию, а не предмет.
    """
    words = _words_in_order(caption)
    return words[0] if words else None


def _caption_matches_query(caption: str, query_words: set[str]) -> bool:
    """
    Подпись должна упомянуть ВСЕ заметные слова запроса, а не только
    большинство — иначе «деревянная лошадь» проходит по одному «horse» в
    подписи «a white horse grazing in the woods», хотя «wooden» там нет и
    в помине. Строже: «dolphin on horseback on bear» и «wooden horse» дают
    пустую выдачу (и генерацию), «dog»/«a brown bear in a forest» — проходят.

    Цена — запросы с деталью, которую BLIP не упомянул («красная машина»,
    если подпись просто «a car on the street»), тоже уйдут в генерацию. Это
    осознанный выбор в пользу точности, а не полноты.
    """
    if not caption or not query_words:
        return True  # без подписи или без ключевых слов запроса — не отсеиваем
    matched = query_words & _content_words(caption)
    return matched == query_words


def filter_by_caption(hits: list[SearchHit], used_query: str) -> list[SearchHit]:
    query_words = _content_words(used_query)
    return [hit for hit in hits if _caption_matches_query(hit.caption, query_words)]


def caption_query_image(query_image) -> str | None:
    """Подпись самой картинки-образца через BLIP — None, если не удалось (BLIP недоступен/упал)."""
    try:
        from core.captioner import Captioner

        return Captioner.get().caption_one(query_image)
    except Exception as e:  # noqa: BLE001 — подпись необязательна, поиск не должен падать из-за неё
        print(f"[подпись картинки-запроса не удалась] {e}")
        return None


def filter_by_image_caption(hits: list[SearchHit], query_caption: str | None) -> list[SearchHit]:
    """
    Фильтрует по ГЛАВНОМУ предмету картинки-запроса, а не по всей сцене
    целиком: BLIP подписывает и лампу, и стол, и цветы рядом с ней одной
    строкой, а искать «по образцу» человек хочет обычно предмет, а не точную
    композицию. Поэтому здесь — только первое слово подписи (_main_word), в
    отличие от текстового поиска (filter_by_caption), где нужны все слова,
    раз их выбрал сам человек.
    """
    if not query_caption:
        return hits  # подпись не удалась — не отсеиваем, лучше прежнее поведение
    main_word = _main_word(query_caption)
    if not main_word:
        return hits
    return [hit for hit in hits if _caption_matches_query(hit.caption, {main_word})]


# --------------------------------------------------------------------------
# Темы для вкладки «Фильмы и музыка» (web/routers/media.py) — те же источники,
# что и у ленты (web/feed.py): недавние запросы и то, что человек лайкнул,
# добавил в избранное или просмотрел. Разница в том, что ленте нужны похожие
# ФОТО (эмбеддинг), а здесь — просто английские слова-темы для OMDb/Last.fm.
# --------------------------------------------------------------------------

# Цвет предмета на фото — не тема для подбора фильма или музыки («red»/«yellow»
# ничего не говорят про жанр или настроение), а в подписях BLIP встречается
# почти в каждой второй строке («a red car», «a yellow submarine»). В отличие
# от _STOPWORDS эти слова НЕ трогают поиск (там цвет — осмысленная деталь
# запроса), поэтому список отдельный и используется только здесь.
_THEME_SKIP_WORDS = {
    "red", "yellow", "blue", "green", "black", "white", "brown", "orange",
    "pink", "purple", "gray", "grey", "beige", "golden", "silver", "dark", "light",
}


def recent_theme_keywords(user_id: str, limit: int = 6) -> list[str]:
    """
    Различимые ключевые слова, самые свежие впереди. Запросы идут первыми:
    их выбрал сам человек, а не BLIP по случайной детали сцены на фото.
    """
    databases = {d["id"]: d for d in db.list_databases(user_id)}
    seen: set[str] = set()
    words: list[str] = []

    def add_words(text: str) -> None:
        for w in _words_in_order(text):
            if w in _THEME_SKIP_WORDS:
                continue
            if w not in seen:
                seen.add(w)
                words.append(w)

    for row in db.recent_queries(user_id, limit=8):
        add_words(row["used_query"])

    for row in db.recent_interacted_photos(user_id, limit=15):
        database = databases.get(row["database_id"])
        if database is None:
            continue
        photo = store_for(database).get_photo(row["photo_id"])
        if photo and photo.caption:
            add_words(photo.caption)

    return words[:limit]


def generated_hit_out(database: dict, photo_id: str) -> SearchHitOut:
    """Единственный результат, когда обычный поиск ничего не нашёл и подключилась генерация."""
    base = f"/api/databases/{database['id']}/photos"
    return SearchHitOut(
        photo_id=photo_id, database_id=database["id"], score=1.0,
        thumb_url=f"{base}/{photo_id}/thumb", file_url=f"{base}/{photo_id}/file", ai_generated=True,
    )


# --------------------------------------------------------------------------
# Генерация фото, когда поиск ничего не нашёл (YandexART)
# --------------------------------------------------------------------------

def generate_fallback_photo(database: dict, user_id: str, prompt: str) -> dict | None:
    """
    Пробует сгенерировать снимок по тексту запроса и сохранить его в базу —
    следующий такой же запрос найдёт его обычным поиском, не тратя квоту API
    заново (ключ общий на всю команду, см. .env.example).

    Ничего не бросает: недоступный ключ, исчерпанный дневной лимит или сбой
    самого YandexART должны молча вернуть None — отсутствие результата не
    может ронять поиск, который до этого работал штатно.
    """
    settings = get_settings()
    prompt = prompt.strip()
    if not settings.photo_generation_enabled or database.get("read_only") or not prompt:
        return None
    if db.count_generations_today(user_id) >= settings.yandex_generations_per_user_day:
        return None

    try:
        image = yandex_art.generate_image(
            prompt, api_key=settings.yandex_api_key, folder_id=settings.yandex_folder_id,
        )
    except yandex_art.YandexArtError as e:
        print(f"[генерация YandexART не удалась] {e}")
        return None

    staging = tmp_dir(database, f"gen-{db.new_id()}")
    name = f"{db.new_id()}.jpg"
    (staging / name).write_bytes(image.content)
    try:
        store = store_for(database)
        outcome = index_files(store, [staging / name], None, {name: name})
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        sync_stats(database["id"], store_for(database))

    # запись в лог — независимо от исхода индексации: сам вызов API уже потратил
    # квоту, и повторная попытка на тот же запрос не должна быть бесплатной
    db.log_generation(user_id)

    photo_id = outcome.get("photo_id")
    if not photo_id:
        return None
    db.mark_ai_generated(database["id"], photo_id, prompt)
    return {"photo_id": photo_id}
