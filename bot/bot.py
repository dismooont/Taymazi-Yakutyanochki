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
  фото             — попадает в базу чата и сразу показывает похожие на него;
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
import logging
import os
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BufferedInputFile, InputMediaPhoto, Message

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

# Без лога бот — чёрный ящик: на жалобу «не отвечает» смотреть было бы нечего.
# Пишем, что пришло и чем закончилось, но без текста запросов и без содержимого
# чатов: в логе не должно оседать то, что люди присылают друг другу.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# httpx на уровне INFO пишет строку на каждый запрос к API и топит в шуме
# собственные записи бота — оставляем от него только предупреждения.
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger("bot")


def _where(message: Message) -> str:
    """Куда отвечаем — тип чата и его id, без названия и участников."""
    return f"{message.chat.type}:{message.chat.id}"


HELP = (
    "🔎 <b>Поиск по фотографиям</b>\n\n"
    "Я запоминаю присланные сюда снимки и нахожу их по описанию.\n"
    "Пришлите фотографию — добавлю её и покажу похожие из этого чата.\n\n"
    "<code>/find рыжий кот на подоконнике</code> — искать в этом чате\n"
    "<code>/demo красный автобус</code> — искать в готовой подборке MS COCO\n"
    "/stats — сколько снимков накопилось\n\n"
    "В личке можно просто написать, что искать, без команды.\n"
    "Запрос можно писать по-русски — я переведу его сам."
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
            log.warning("/start %s -> отказ: %s", _where(message), e.detail)
            await message.answer(f"Не получилось: {e.detail}")
            return
        log.info("/start %s -> база %s, создана=%s",
                 _where(message), chat["database_id"][:8], chat["created"])

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
            log.warning("фото %s -> отказ: %s", _where(message), e.detail)
            # в группе о каждой неудаче не сообщаем, чтобы не засорять чат
            if message.chat.type in PRIVATE:
                await message.reply(f"Не добавил: {e.detail}")
            return
        log.info("фото %s -> добавлено=%s, всего=%s",
                 _where(message), result["added"], result["photos_count"])

        # Присланное фото — это ещё и запрос «покажи похожее». Ищем по нему всегда,
        # в том числе в группе: там ответ нужен не меньше, а команду для картинки
        # не напишешь. Эмбеддинг заново не считается — API берёт вектор из индекса.
        photo_id = result.get("photo_id")
        if not photo_id:
            if message.chat.type in PRIVATE and result["skipped"]:
                await message.reply(f"Пропустил: {result['skipped'][0][1]}.")
            return

        try:
            similar = await api.similar(message.chat.id, photo_id, TOP_K)
        except ApiError as e:
            log.warning("похожие %s -> отказ: %s", _where(message), e.detail)
            if message.chat.type in PRIVATE:
                await message.reply("Добавил в базу.")
            return

        hits = similar["results"]
        log.info("похожие %s -> найдено %s", _where(message), len(hits))
        if not hits:
            # первое фото в базе: сравнивать не с чем, и это не ошибка
            await message.reply("Добавил в базу. Пока это единственный снимок — сравнить не с чем.")
            return

        await message.reply(f"Добавил в базу. Похожие ({len(hits)}):")
        await _send_album(message, hits, lambda pid: api.photo_bytes(message.chat.id, pid))

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

    @dispatcher.message(Command("demo"))
    async def on_demo(message: Message, command: CommandObject) -> None:
        """
        Поиск по общей демо-базе MS COCO — 5022 размеченных снимка.

        Команда одноразовая, а не переключатель режима: с режимом человек рано или
        поздно забудет, где находится, и удивится, почему поиск отвечает чужими
        фотографиями.
        """
        demo = await api.demo_info()
        if demo is None:
            await message.answer("Демо-база не подключена на этом сервере.")
            return

        query = (command.args or "").strip()
        if not query:
            await message.answer(
                f"Демо-база: {demo['photos_count']} фотографий MS COCO.\n"
                f"Что найти? Например: <code>/demo красный автобус</code>"
            )
            return

        await _search_and_send(
            message, query,
            search=lambda: api.search_demo(query, TOP_K),
            fetch=api.demo_photo_bytes,
            source="демо",
        )

    async def _answer(message: Message, query: str) -> None:
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        if chat["photos_count"] == 0:
            await message.answer(
                "В базе пока нет фотографий — пришлите несколько.\n"
                "Или посмотрите, как это работает, на готовой подборке: "
                "<code>/demo красный автобус</code>"
            )
            return

        await _search_and_send(
            message, query,
            search=lambda: api.search(message.chat.id, query, TOP_K),
            fetch=lambda photo_id: api.photo_bytes(message.chat.id, photo_id),
            source="чат",
        )

    async def _search_and_send(message: Message, query: str, search, fetch, source: str) -> None:
        """Общая часть для поиска по базе чата и по демо-базе: запрос, выдача, отправка."""
        try:
            found = await search()
        except ApiError as e:
            log.warning("поиск(%s) %s -> отказ: %s", source, _where(message), e.detail)
            await message.answer(f"Поиск не удался: {e.detail}")
            return
        # длину запроса пишем, сам запрос — нет: это личные данные
        log.info("поиск(%s) %s -> %s символов, найдено %s",
                 source, _where(message), len(query), len(found["results"]))

        hits = found["results"]
        if not hits:
            await message.answer("Ничего не нашлось.")
            return

        used = found["used_query"]
        note = f"«{query}» → <code>{used}</code>" if used != query else f"«{query}»"
        where = " в демо-базе" if source == "демо" else ""
        await message.answer(f"{note}\nНашёл {len(hits)}{where}:")
        await _send_album(message, hits, fetch)

    async def _send_album(message: Message, hits: list, fetch) -> None:
        """
        Отправляет найденное одним альбомом, а не отдельными сообщениями: пять
        картинок подряд забивают весь экран, а альбом Telegram показывает сеткой.
        Близость подписана у каждой — она разная у разных мест выдачи.
        """
        media, missing = [], 0
        for place, hit in enumerate(hits, start=1):
            try:
                data = await fetch(hit["photo_id"])
            except ApiError:
                missing += 1
                continue
            media.append(InputMediaPhoto(
                media=BufferedInputFile(data, filename=f"{hit['photo_id']}.jpg"),
                caption=f"{place}. близость {hit['score']:.3f}",
            ))

        if not media:
            await message.answer("Файлы найденных снимков недоступны.")
            return
        if len(media) == 1:
            # альбом из одного элемента Telegram не принимает
            await message.answer_photo(media[0].media, caption=media[0].caption)
        else:
            await message.answer_media_group(media)
        if missing:
            await message.answer(f"Ещё {missing} снимков не удалось прочитать с диска.")

    return dispatcher


def _proxy_for_this_host() -> str:
    """
    Адрес прокси, пригодный для того места, где бот действительно запущен.

    В .env он обычно записан для Docker — host.docker.internal. Вне контейнера
    такого имени не существует, и бот молча не достучался бы до Telegram. Один
    и тот же .env должен работать в обоих режимах, поэтому при запуске на машине
    подставляем localhost.
    """
    proxy = os.environ.get("TELEGRAM_PROXY", "").strip()
    in_docker = Path("/.dockerenv").exists()
    if proxy and not in_docker and "host.docker.internal" in proxy:
        local = proxy.replace("host.docker.internal", "127.0.0.1")
        log.info("Запуск вне Docker: прокси %s -> %s", proxy, local)
        return local
    return proxy


async def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("Не задан TELEGRAM_BOT_TOKEN (см. .env.example)")
    service_token = os.environ.get("SERVICE_TOKEN", "").strip()
    if not service_token:
        sys.exit("Не задан SERVICE_TOKEN — тот же, что у веб-приложения")

    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
    api = SearchApi(api_url, service_token)

    proxy = _proxy_for_this_host()
    bot = Bot(
        token=token,
        session=AiohttpSession(proxy=proxy) if proxy else None,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    log.info("Бот запущен, API: %s", api_url)
    try:
        await build_dispatcher(api).start_polling(bot)
    finally:
        await api.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
