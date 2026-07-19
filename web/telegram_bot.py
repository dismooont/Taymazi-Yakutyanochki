"""
Telegram-бот: своя база у каждого чата.

Бот живёт внутри веб-процесса, а не отдельным контейнером, по трём причинам:
  * веса CLIP занимают ~600 МБ, и второй процесс означал бы вторую копию в памяти;
  * два процесса писали бы в одни и те же файлы FAISS-индекса наперегонки;
  * Telegram отдаёт обновления только одному потребителю getUpdates на токен.

Что делает бот:
  /start в личке   — заводит личную базу пользователя (она же видна ему на сайте);
  /start в группе  — заводит базу чата, после чего КАЖДОЕ новое фото индексируется;
  фото             — попадает в базу того чата, где отправлено;
  текст в личке    — поиск; в группе — поиск по команде /find, чтобы бот не лез
                     в каждое сообщение.

ВАЖНО про групповые чаты: по умолчанию у бота включён режим приватности, и Telegram
показывает ему только команды и ответы на его сообщения. Чтобы бот видел присылаемые
в группу фотографии, у @BotFather нужно выполнить /setprivacy -> Disable.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from web import db
from web.config import get_settings
from web.stores import create_store, store_for, sync_stats

TOP_K = 5
PRIVATE_CHATS = {"private"}


# --------------------------------------------------------------------------
# Аккаунты и базы
# --------------------------------------------------------------------------

def _account_for(message: Message) -> dict[str, Any]:
    """
    Аккаунт, соответствующий отправителю. Тот же самый, в который человек попадёт,
    если войдёт на сайт через Telegram, — благодаря общей таблице identities.
    """
    telegram_id = str(message.from_user.id)
    user = db.get_user_by_identity("telegram", telegram_id)
    if user is None:
        display = " ".join(
            part for part in (message.from_user.first_name, message.from_user.last_name) if part
        )
        user = db.create_user(
            login=None, display_name=display or f"tg{telegram_id}", password_hash=None
        )
        db.link_identity("telegram", telegram_id, user["id"])
    return user


def _database_for(message: Message) -> dict[str, Any] | None:
    """База этого чата или None, если бот здесь ещё не запускали."""
    return db.get_database_by_chat(str(message.chat.id))


def _create_database_for(message: Message, owner: dict[str, Any]) -> dict[str, Any]:
    name = (
        "Личный архив"
        if message.chat.type in PRIVATE_CHATS
        else (message.chat.title or f"Чат {message.chat.id}")
    )
    database = db.create_database(
        owner["id"], name, kind="chat", telegram_chat_id=str(message.chat.id)
    )
    create_store(owner["id"], database["id"])
    return sync_stats(database["id"], store_for(database))


# --------------------------------------------------------------------------
# Обработчики
# --------------------------------------------------------------------------

def build_dispatcher() -> Dispatcher:
    dispatcher = Dispatcher()

    @dispatcher.message(CommandStart())
    async def on_start(message: Message) -> None:
        user = await asyncio.to_thread(_account_for, message)
        db.remember_chat_member(message.chat.id, user["id"])

        database = await asyncio.to_thread(_database_for, message)
        if database is None:
            database = await asyncio.to_thread(_create_database_for, message, user)
            where = "этого чата" if message.chat.type not in PRIVATE_CHATS else "вашего архива"
            await message.answer(
                f"База {where} создана. Теперь я запоминаю каждое присланное сюда фото "
                f"и умею искать по описанию.\n\n"
                f"Найти: <code>/find рыжий кот на подоконнике</code>\n"
                f"Сколько накопилось: /stats\n\n"
                f"Те же снимки доступны на сайте — войдите через Telegram."
            )
        else:
            await message.answer(
                f"База «{database['name']}» уже работает: "
                f"{database['photos_count']} фото. Ищите командой /find."
            )

    @dispatcher.message(Command("stats"))
    async def on_stats(message: Message) -> None:
        database = await asyncio.to_thread(_database_for, message)
        if database is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        fresh = await asyncio.to_thread(
            lambda: sync_stats(database["id"], store_for(database))
        )
        megabytes = (fresh["photos_bytes"] + fresh["index_bytes"]) / 1024 / 1024
        await message.answer(
            f"«{fresh['name']}»: {fresh['photos_count']} фото, {megabytes:.1f} МБ."
        )

    @dispatcher.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        """
        Фото индексируется, только если в чате уже запускали /start. Пока команду
        не выполнили, бот молчит и ничего не сохраняет: складывать чужие снимки
        на сервер, никого не спросив, нельзя.
        """
        database = await asyncio.to_thread(_database_for, message)
        if database is None:
            return

        user = await asyncio.to_thread(_account_for, message)
        db.remember_chat_member(message.chat.id, user["id"])

        photo = message.photo[-1]  # последний элемент — самое крупное превью
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{photo.file_unique_id}.jpg"
            await bot.download(photo, destination=path)
            added = await asyncio.to_thread(_add_photo, database, path)

        if added and message.chat.type in PRIVATE_CHATS:
            # в группе на каждое фото не отвечаем — бот превратился бы в спам
            await message.reply("Добавил в базу.")

    @dispatcher.message(Command("find"))
    async def on_find(message: Message, command: Any) -> None:
        query = (command.args or "").strip()
        if not query:
            await message.answer("Что искать? Например: <code>/find собака в снегу</code>")
            return
        await _answer_search(message, query)

    @dispatcher.message(F.text, F.chat.type.in_(PRIVATE_CHATS))
    async def on_private_text(message: Message) -> None:
        """В личке любое сообщение — это запрос: команда /find тут была бы лишней."""
        await _answer_search(message, message.text.strip())

    async def _answer_search(message: Message, query: str) -> None:
        database = await asyncio.to_thread(_database_for, message)
        if database is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        if database["photos_count"] == 0:
            await message.answer("В базе пока нет фотографий — пришлите несколько.")
            return

        used_query, hits = await asyncio.to_thread(_search, database, query)
        if not hits:
            await message.answer("Ничего не нашлось.")
            return

        note = f"«{query}» → <code>{used_query}</code>" if used_query != query else f"«{query}»"
        await message.answer(f"{note}\nНашёл {len(hits)}:")
        for hit in hits:
            try:
                data = Path(hit.path).read_bytes()
            except OSError:
                continue
            await message.answer_photo(
                BufferedInputFile(data, filename=f"{hit.photo_id}.jpg"),
                caption=f"близость {hit.score:.3f}",
            )

    return dispatcher


def _add_photo(database: dict, path: Path) -> bool:
    store = store_for(database)
    result = store.add_photos([path])
    sync_stats(database["id"], store)
    return result.added_count > 0


def _search(database: dict, query: str):
    store = store_for(database)
    return store.search_text(query, top_k=TOP_K)


# --------------------------------------------------------------------------
# Запуск
# --------------------------------------------------------------------------

async def run_bot(stop: asyncio.Event) -> None:
    """Длинный опрос до остановки приложения."""
    settings = get_settings()
    session = None
    proxy = settings.telegram_proxy
    if proxy:
        session = AiohttpSession(proxy=proxy)

    bot = Bot(
        token=settings.telegram_bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    dispatcher = build_dispatcher()

    polling = asyncio.create_task(dispatcher.start_polling(bot, handle_signals=False))
    try:
        await stop.wait()
    finally:
        await dispatcher.stop_polling()
        polling.cancel()
        await bot.session.close()
