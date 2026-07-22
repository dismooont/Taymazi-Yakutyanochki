"""
Telegram-бот: своя база фотографий у каждого чата.

Бот — тонкий клиент API (bot/client.py). Он не загружает CLIP и не трогает файлы
индекса: всё, что он делает с базой, выполняет API, поэтому квоты, очередь задач
и запрет на изменение демо-базы действуют одинаково и в чате, и на сайте.
А модель в системе существует ровно в одном экземпляре.

Управление — кнопками, а не командами. Внизу постоянное меню (Найти, Демо,
Статистика, Архив, Помощь), а под каждым найденным снимком — действия сайта:
удалить, подпись, похожие. Команды (/find, /demo, /stats) сохранены как запасной
путь и для групп.

ВАЖНО про групповые чаты: по умолчанию у бота включён режим приватности, и Telegram
показывает ему только команды. Чтобы бот видел присылаемые в группу фотографии,
у @BotFather нужно выполнить /setprivacy -> Disable.

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
from aiogram.filters import Command, CommandObject, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

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

# Тексты кнопок нижнего меню. Они же — то, что приходит как обычный текст, когда
# кнопку нажали, поэтому вынесены в константы: по ним отличаем нажатие меню от
# поискового запроса.
BTN_FIND = "🔎 Найти"
BTN_DEMO = "🎞 Демо"
BTN_STATS = "📊 Статистика"
BTN_EXPORT = "📦 Архив"
BTN_HELP = "❓ Помощь"
MENU_TEXTS = {BTN_FIND, BTN_DEMO, BTN_STATS, BTN_EXPORT, BTN_HELP}

MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_FIND), KeyboardButton(text=BTN_DEMO)],
        [KeyboardButton(text=BTN_STATS), KeyboardButton(text=BTN_EXPORT)],
        [KeyboardButton(text=BTN_HELP)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Опишите, что искать",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")


class Flow(StatesGroup):
    """Состояния диалога: бот ждёт от пользователя ввод после нажатия кнопки."""

    waiting_query = State()    # ждём поисковый запрос
    waiting_demo = State()     # ждём запрос к демо-базе
    waiting_caption = State()  # ждём текст подписи для конкретного снимка


HELP = (
    "🔎 <b>Поиск по фотографиям</b>\n\n"
    "Я запоминаю присланные сюда снимки и нахожу их по описанию.\n"
    "Пришлите фотографию — добавлю её и покажу похожие.\n"
    "Пришлите zip-архив — добавлю сразу все фото из него.\n\n"
    "Кнопки внизу:\n"
    f"{BTN_FIND} — искать в этом чате\n"
    f"{BTN_DEMO} — искать в готовой подборке MS COCO\n"
    f"{BTN_STATS} — сколько снимков накопилось\n"
    f"{BTN_EXPORT} — скачать всю базу одним zip\n\n"
    "Под каждым найденным снимком — 🗑 удалить, ✏️ подпись, 🔍 похожие.\n"
    "Запрос можно писать по-русски — переведу сам."
)


def _where(message: Message) -> str:
    return f"{message.chat.type}:{message.chat.id}"


def _sender(message: Message) -> tuple[int, str]:
    user = message.from_user
    name = " ".join(part for part in (user.first_name, user.last_name) if part)
    return user.id, name or (user.username or f"tg{user.id}")


def _actions(photo_id: str) -> InlineKeyboardMarkup:
    """Кнопки действий под снимком из базы чата — то же, что доступно на сайте."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Удалить", callback_data=f"del:{photo_id}"),
        InlineKeyboardButton(text="✏️ Подпись", callback_data=f"cap:{photo_id}"),
        InlineKeyboardButton(text="🔍 Похожие", callback_data=f"sim:{photo_id}"),
    ]])


def _menu_for(message: Message) -> ReplyKeyboardMarkup | None:
    """Нижнее меню показываем в личке; в группе общая клавиатура была бы шумом."""
    return MENU if message.chat.type in PRIVATE else None


def build_dispatcher(api: SearchApi) -> Dispatcher:
    dispatcher = Dispatcher(storage=MemoryStorage())

    # ------------------------------------------------------------------
    # Старт и справка
    # ------------------------------------------------------------------

    @dispatcher.message(CommandStart())
    async def on_start(message: Message, state: FSMContext) -> None:
        await state.clear()
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
            text = f"База {where} создана. Теперь я запоминаю каждое присланное сюда фото.\n\n" + HELP
        else:
            text = f"База «{chat['name']}» уже работает: {chat['photos_count']} фото.\n\n" + HELP
        await message.answer(text, reply_markup=_menu_for(message))

    @dispatcher.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(HELP, reply_markup=_menu_for(message))

    @dispatcher.message(F.text == BTN_HELP)
    async def on_help_btn(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(HELP, reply_markup=_menu_for(message))

    # ------------------------------------------------------------------
    # Статистика и экспорт
    # ------------------------------------------------------------------

    @dispatcher.message(Command("stats"))
    @dispatcher.message(F.text == BTN_STATS)
    async def on_stats(message: Message, state: FSMContext) -> None:
        await state.clear()
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        megabytes = chat["total_bytes"] / 1024 / 1024
        line = f"«{chat['name']}»: {chat['photos_count']} фото, {megabytes:.1f} МБ."
        covered = chat.get("captions_count", 0)
        if covered:
            line += f"\nРазмечено подписями: {covered} из {chat['photos_count']}."
        await message.answer(line)

    @dispatcher.message(F.text == BTN_EXPORT)
    async def on_export(message: Message, state: FSMContext) -> None:
        await state.clear()
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        if chat["photos_count"] == 0:
            await message.answer("В базе пока нет снимков — нечего выгружать.")
            return
        await message.answer("Собираю архив…")
        try:
            data = await api.export_bytes(message.chat.id)
        except ApiError as e:
            await message.answer(f"Не удалось собрать архив: {e.detail}")
            return
        await message.answer_document(
            BufferedInputFile(data, filename=f"{chat['name']}.zip"),
            caption=f"База «{chat['name']}» — {chat['photos_count']} фото.",
        )

    # ------------------------------------------------------------------
    # Поиск: по кнопке спрашиваем запрос, следующее сообщение — запрос
    # ------------------------------------------------------------------

    @dispatcher.message(F.text == BTN_FIND)
    async def on_find_btn(message: Message, state: FSMContext) -> None:
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        await state.set_state(Flow.waiting_query)
        await message.answer("Что искать? Опишите снимок — например «рыжий кот на подоконнике».")

    @dispatcher.message(F.text == BTN_DEMO)
    async def on_demo_btn(message: Message, state: FSMContext) -> None:
        demo = await api.demo_info()
        if demo is None:
            await message.answer("Демо-база не подключена на этом сервере.")
            return
        await state.set_state(Flow.waiting_demo)
        await message.answer(
            f"Демо-база: {demo['photos_count']} фотографий MS COCO.\nЧто найти?"
        )

    @dispatcher.message(Command("find"))
    async def on_find_cmd(message: Message, command: CommandObject, state: FSMContext) -> None:
        query = (command.args or "").strip()
        if not query:
            await message.answer("Что искать? Например: <code>/find собака в снегу</code>")
            return
        await state.clear()
        await _answer(message, query)

    @dispatcher.message(Command("demo"))
    async def on_demo_cmd(message: Message, command: CommandObject, state: FSMContext) -> None:
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
        await state.clear()
        await _search_demo(message, query)

    @dispatcher.message(StateFilter(Flow.waiting_query), F.text)
    async def on_query_text(message: Message, state: FSMContext) -> None:
        await state.clear()
        await _answer(message, message.text.strip())

    @dispatcher.message(StateFilter(Flow.waiting_demo), F.text)
    async def on_demo_text(message: Message, state: FSMContext) -> None:
        await state.clear()
        await _search_demo(message, message.text.strip())

    # ------------------------------------------------------------------
    # Ввод подписи (после кнопки ✏️ под снимком)
    # ------------------------------------------------------------------

    @dispatcher.message(StateFilter(Flow.waiting_caption), F.text)
    async def on_caption_text(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        await state.clear()
        photo_id = data.get("photo_id")
        caption = message.text.strip()
        try:
            result = await api.set_caption(message.chat.id, photo_id, caption)
        except ApiError as e:
            await message.answer(f"Не удалось сохранить подпись: {e.detail}")
            return
        if result["caption"]:
            await message.answer(f"✏️ Подпись сохранена: «{result['caption']}»")
        else:
            await message.answer("Подпись снята.")

    # ------------------------------------------------------------------
    # Фото: добавляем и показываем похожие
    # ------------------------------------------------------------------

    @dispatcher.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        if await api.chat_info(message.chat.id) is None:
            return

        user_id, name = _sender(message)
        photo = message.photo[-1]
        buffer = io.BytesIO()
        await bot.download(photo, destination=buffer)

        try:
            result = await api.add_photo(
                message.chat.id, f"{photo.file_unique_id}.jpg", buffer.getvalue(), user_id, name
            )
        except ApiError as e:
            log.warning("фото %s -> отказ: %s", _where(message), e.detail)
            if message.chat.type in PRIVATE:
                await message.reply(f"Не добавил: {e.detail}")
            return
        log.info("фото %s -> добавлено=%s, всего=%s",
                 _where(message), result["added"], result["photos_count"])

        photo_id = result.get("photo_id")
        if not photo_id:
            if message.chat.type in PRIVATE and result["skipped"]:
                await message.reply(f"Пропустил: {result['skipped'][0][1]}.")
            return

        try:
            similar = await api.similar(message.chat.id, photo_id, TOP_K)
        except ApiError:
            await message.reply("Добавил в базу.")
            return

        hits = similar["results"]
        if not hits:
            await message.reply("Добавил в базу. Пока это единственный снимок — сравнить не с чем.")
            return
        await message.reply(f"Добавил в базу. Похожие ({len(hits)}):")
        await _send_hits(message, hits)

    # ------------------------------------------------------------------
    # Импорт группы фото из zip-архива (как «загрузить архив» на сайте)
    # ------------------------------------------------------------------

    @dispatcher.message(F.document)
    async def on_document(message: Message, bot: Bot) -> None:
        doc = message.document
        name = (doc.file_name or "").lower()
        is_zip = name.endswith(".zip") or (doc.mime_type or "") in (
            "application/zip", "application/x-zip-compressed",
        )
        if not is_zip:
            if message.chat.type in PRIVATE:
                await message.reply("Пришлите zip-архив с фотографиями — добавлю все сразу.")
            return
        if await api.chat_info(message.chat.id) is None:
            if message.chat.type in PRIVATE:
                await message.reply("Здесь ещё нет базы. Начните с /start.")
            return

        buffer = io.BytesIO()
        try:
            await bot.download(doc, destination=buffer)
        except Exception:
            await message.reply(
                "Не смог скачать архив. Возможно, он больше 20 МБ — таков лимит Telegram для ботов."
            )
            return

        try:
            started = await api.import_archive(
                message.chat.id, doc.file_name or "archive.zip", buffer.getvalue()
            )
        except ApiError as e:
            await message.reply(f"Архив не принят: {e.detail}")
            return

        job_id, count = started["job_id"], started["count"]
        log.info("импорт %s -> задача %s, фото в архиве %s", _where(message), job_id[:8], count)
        status = await message.reply(f"Архив принят: {count} фото. Добавляю…")

        # опрашиваем задачу и обновляем то же сообщение прогрессом
        for _ in range(300):  # шаг 2 с -> около 10 минут терпения
            await asyncio.sleep(2)
            try:
                job = await api.job(message.chat.id, job_id)
            except ApiError:
                break
            if job["status"] in ("done", "error"):
                done_ok = job["status"] == "done"
                text = job.get("message") or ("Импорт завершён." if done_ok else "Ошибка импорта.")
                await _safe_edit(status, ("Готово. " if done_ok else "Не удалось: ") + text)
                return
            total = job["progress_total"]
            if total:
                await _safe_edit(status, f"Добавляю: {job['progress_done']}/{total}…")
        await _safe_edit(status, "Импорт идёт дольше обычного — загляните позже через 📊 Статистика.")

    # ------------------------------------------------------------------
    # Свободный текст в личке (без состояния) — тоже поиск
    # ------------------------------------------------------------------

    @dispatcher.message(F.text, ~F.text.startswith("/"), F.chat.type.in_(PRIVATE))
    async def on_private_text(message: Message) -> None:
        # нажатия меню отлавливаются обработчиками выше; сюда попадает произвольный
        # текст, и в личке это запрос — команда /find была бы лишней
        if message.text.strip() in MENU_TEXTS:
            return
        await _answer(message, message.text.strip())

    @dispatcher.message(F.text.startswith("/"), F.chat.type.in_(PRIVATE))
    async def on_unknown_command(message: Message) -> None:
        await message.answer("Не знаю такой команды.\n\n" + HELP, reply_markup=_menu_for(message))

    # ------------------------------------------------------------------
    # Колбэки инлайн-кнопок под снимками
    # ------------------------------------------------------------------

    @dispatcher.callback_query(F.data.startswith("del:"))
    async def on_delete(callback: CallbackQuery) -> None:
        photo_id = callback.data.split(":", 1)[1]
        try:
            await api.delete_photo(callback.message.chat.id, photo_id)
        except ApiError as e:
            await callback.answer(f"Не удалил: {e.detail}", show_alert=True)
            return
        await callback.answer("Удалено")
        try:
            await callback.message.edit_caption(caption="🗑 Снимок удалён", reply_markup=None)
        except Exception:
            pass

    @dispatcher.callback_query(F.data.startswith("cap:"))
    async def on_caption_btn(callback: CallbackQuery, state: FSMContext) -> None:
        photo_id = callback.data.split(":", 1)[1]
        await state.set_state(Flow.waiting_caption)
        await state.update_data(photo_id=photo_id)
        await callback.answer()
        await callback.message.answer(
            "Введите подпись для снимка (или «-», чтобы снять):"
        )

    @dispatcher.callback_query(F.data.startswith("sim:"))
    async def on_similar_btn(callback: CallbackQuery) -> None:
        photo_id = callback.data.split(":", 1)[1]
        await callback.answer()
        try:
            similar = await api.similar(callback.message.chat.id, photo_id, TOP_K)
        except ApiError as e:
            await callback.message.answer(f"Не удалось: {e.detail}")
            return
        hits = [h for h in similar["results"] if h["photo_id"] != photo_id]
        if not hits:
            await callback.message.answer("Похожих не нашлось.")
            return
        await callback.message.answer(f"Похожие ({len(hits)}):")
        await _send_hits(callback.message, hits)

    # ------------------------------------------------------------------
    # Общие помощники поиска и отправки
    # ------------------------------------------------------------------

    async def _answer(message: Message, query: str) -> None:
        chat = await api.chat_info(message.chat.id)
        if chat is None:
            await message.answer("Здесь ещё нет базы. Начните с /start.")
            return
        if chat["photos_count"] == 0:
            await message.answer(
                "В базе пока нет фотографий — пришлите несколько.\n"
                f"Или посмотрите на готовой подборке: кнопка {BTN_DEMO}."
            )
            return
        try:
            found = await api.search(message.chat.id, query, TOP_K)
        except ApiError as e:
            await message.answer(f"Поиск не удался: {e.detail}")
            return
        log.info("поиск(чат) %s -> %s символов, найдено %s",
                 _where(message), len(query), len(found["results"]))
        hits = found["results"]
        if not hits:
            await message.answer("Ничего похожего не нашлось.")
            return
        await message.answer(_query_note(query, found["used_query"], len(hits)))
        await _send_hits(message, hits)

    async def _search_demo(message: Message, query: str) -> None:
        try:
            found = await api.search_demo(query, TOP_K)
        except ApiError as e:
            await message.answer(f"Поиск не удался: {e.detail}")
            return
        hits = found["results"]
        if not hits:
            await message.answer("В демо-базе ничего похожего не нашлось.")
            return
        note = _query_note(query, found["used_query"], len(hits)) + " в демо-базе:"
        await message.answer(note)
        # демо-база только для чтения: удалять/подписывать нечего, поэтому альбом
        await _send_album(message, hits, api.demo_photo_bytes)

    def _query_note(query: str, used: str, n: int) -> str:
        head = f"«{query}» → <code>{used}</code>" if used and used != query else f"«{query}»"
        return f"{head}\nНашёл {n}:"

    async def _safe_edit(message: Message, text: str) -> None:
        # правка тем же текстом или устаревшего сообщения бросает исключение —
        # для индикатора прогресса это не ошибка, а норма
        try:
            await message.edit_text(text)
        except Exception:
            pass

    async def _send_hits(message: Message, hits: list) -> None:
        """
        Результаты по базе чата — по одному снимку с кнопками действий под каждым.
        Альбомом их слать нельзя: Telegram не разрешает кнопки под медиа-группой,
        а именно кнопки (удалить/подпись/похожие) и есть суть паритета с сайтом.
        """
        sent = 0
        for place, hit in enumerate(hits, start=1):
            try:
                data = await api.photo_bytes(message.chat.id, hit["photo_id"])
            except ApiError:
                continue
            caption = f"{place}. близость {hit['score']:.3f}"
            if hit.get("caption"):
                caption += f"\n✏️ {hit['caption']}"
            await message.answer_photo(
                BufferedInputFile(data, filename=f"{hit['photo_id']}.jpg"),
                caption=caption,
                reply_markup=_actions(hit["photo_id"]),
            )
            sent += 1
        if not sent:
            await message.answer("Файлы найденных снимков недоступны.")

    async def _send_album(message: Message, hits: list, fetch) -> None:
        """Демо-выдача — альбомом (read-only, кнопки не нужны)."""
        media = []
        for place, hit in enumerate(hits, start=1):
            try:
                data = await fetch(hit["photo_id"])
            except ApiError:
                continue
            media.append(InputMediaPhoto(
                media=BufferedInputFile(data, filename=f"{hit['photo_id']}.jpg"),
                caption=f"{place}. близость {hit['score']:.3f}",
            ))
        if not media:
            await message.answer("Файлы найденных снимков недоступны.")
        elif len(media) == 1:
            await message.answer_photo(media[0].media, caption=media[0].caption)
        else:
            await message.answer_media_group(media)

    return dispatcher


def _proxy_for_this_host() -> str:
    """
    Адрес прокси, пригодный для того места, где бот действительно запущен.
    В .env он обычно записан для Docker (host.docker.internal); вне контейнера
    такого имени нет, поэтому подставляем localhost.
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
