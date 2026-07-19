"""
Telegram-бот: своя база фотографий у каждого чата.

Бот — тонкий клиент API (bot/client.py). Он не загружает CLIP и не трогает файлы
индекса: всё, что он делает с базой, выполняет API, поэтому квоты, очередь задач
и запрет на изменение демо-базы действуют одинаково и в чате, и на сайте.
А модель в системе существует ровно в одном экземпляре.

Что умеет:
  /start в личке   — заводит личную базу (она же видна на сайте после входа
                     через Telegram);
  /start в группе  — заводит базу чата, после чего КАЖДОЕ новое фото индексируется;
  фото             — попадает в базу того чата, где отправлено;
  текст в личке    — поиск; в группе — /find, чтобы бот не лез в каждое сообщение;
  /stats           — сколько снимков накопилось.

ВАЖНО про групповые чаты: по умолчанию у бота включён режим приватности, и Telegram
показывает ему только команды и ответы на его сообщения. Чтобы бот видел присылаемые
в группу фотографии, у @BotFather нужно выполнить /setprivacy -> Disable.

Запуск:
    python -m bot.bot
Нужны переменные окружения (см. .env.example):
    TELEGRAM_BOT_TOKEN  — токен от @BotFather
    API_URL             — адрес API, по умолчанию http://127.0.0.1:8000
    SERVICE_TOKEN       — тот же, что задан веб-приложению
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BufferedInputFile, Message

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.client import ApiError, SearchApi  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

TOP_K = int(os.environ.get("BOT_TOP_K", "5"))
PRIVATE = {"private"}

HELP = (
    "🔎 <b>Поиск по фотографиям чата</b>\n\n"
    "Я запоминаю присланные сюда снимки и нахожу их по описанию.\n\n"
    "<code>/find рыжий кот на подоконнике</code> — найти\n"
    "/stats — сколько накопилось\n\n"
    "В личке можно просто написать, что искать, без команды."
)


def _sender(message: Message) -> tuple[int, str]:
    user = message.from_user
    name = " ".join(part for part in (user.first_name, user.last_name) if part)
    return user.id, name or (user.username or f"tg{user.id}")


def build_dispatcher(api: SearchApi) -> Dispatcher:
    dispatcher = Dispatcher()

    @dispatcher.message(CommandStart())
    async def on_start(message: Message) -> None:
        user_id, name = _sender(message)
        title = (
            "Личный архив"
            if message.chat.type in PRIVATE
            else (message.chat.title or f"Чат {message.chat.id}")
        )
        try:
            chat = await api.start_chat(message.chat.id, user_id, name, title)
        except ApiError as e:
            await message.answer(f"Не получилось: {e.detail}")
            return

        if chat["created"]:
            where = "этого чата" if message.chat.type not in PRIVATE else "вашего архива"
            await message.answer(
                f"База {where} создана. Теперь я запоминаю каждое присланное сюда фото.\n\n"
                + HELP
            )
        else:
            await message.answer(
                f"База «{chat['name']}» уже работает: {chat['photos_count']} фото.\n\n" + HELP
            )

    @dispatcher.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(HELP)

    @dispatcher.message(Command("stats"))
    async def on_stats(message: Message) -> None:
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        megabytes = chat["total_bytes"] / 1024 / 1024
        await message.answer(
            f"«{chat['name']}»: {chat['photos_count']} фото, {megabytes:.1f} МБ."
        )

    @dispatcher.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        """
        Фото индексируется, только если в чате уже выполнили /start. Пока команду
        не дали, бот молчит и ничего не сохраняет: складывать чужие снимки на
        сервер, никого не спросив, нельзя.
        """
        if await api.chat_info(message.chat.id) is None:
            return

        user_id, name = _sender(message)
        photo = message.photo[-1]  # последний элемент — самое крупное превью
        buffer = io.BytesIO()
        await bot.download(photo, destination=buffer)

        try:
            result = await api.add_photo(
                message.chat.id, f"{photo.file_unique_id}.jpg", buffer.getvalue(), user_id, name
            )
        except ApiError as e:
            # в группе о каждой неудаче не сообщаем, чтобы не засорять чат
            if message.chat.type in PRIVATE:
                await message.reply(f"Не добавил: {e.detail}")
            return

        if message.chat.type in PRIVATE:
            if result["added"]:
                await message.reply("Добавил в базу.")
            elif result["skipped"]:
                await message.reply(f"Пропустил: {result['skipped'][0][1]}.")

    @dispatcher.message(Command("find"))
    async def on_find(message: Message, command: CommandObject) -> None:
        query = (command.args or "").strip()
        if not query:
            await message.answer("Что искать? Например: <code>/find собака в снегу</code>")
            return
        await _answer(message, query)

    @dispatcher.message(F.text, F.chat.type.in_(PRIVATE))
    async def on_private_text(message: Message) -> None:
        """В личке любое сообщение — запрос: команда /find тут была бы лишней."""
        await _answer(message, message.text.strip())

    async def _answer(message: Message, query: str) -> None:
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        if chat["photos_count"] == 0:
            await message.answer("В базе пока нет фотографий — пришлите несколько.")
            return

        try:
            found = await api.search(message.chat.id, query, TOP_K)
        except ApiError as e:
            await message.answer(f"Поиск не удался: {e.detail}")
            return

        hits = found["results"]
        if not hits:
            await message.answer("Ничего не нашлось.")
            return

        used = found["used_query"]
        note = f"«{query}» → <code>{used}</code>" if used != query else f"«{query}»"
        await message.answer(f"{note}\nНашёл {len(hits)}:")
        for hit in hits:
            try:
                data = await api.photo_bytes(message.chat.id, hit["photo_id"])
            except ApiError:
                continue
            await message.answer_photo(
                BufferedInputFile(data, filename=f"{hit['photo_id']}.jpg"),
                caption=f"близость {hit['score']:.3f}",
            )

    return dispatcher


async def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("Не задан TELEGRAM_BOT_TOKEN (см. .env.example)")
    service_token = os.environ.get("SERVICE_TOKEN", "").strip()
    if not service_token:
        sys.exit("Не задан SERVICE_TOKEN — тот же, что у веб-приложения")

    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
    api = SearchApi(api_url, service_token)

    proxy = os.environ.get("TELEGRAM_PROXY", "").strip()
    bot = Bot(
        token=token,
        session=AiohttpSession(proxy=proxy) if proxy else None,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    print(f"Бот запущен, API: {api_url}")
    try:
        await build_dispatcher(api).start_polling(bot)
    finally:
        await api.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
