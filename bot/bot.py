"""
Telegram-бот для CLIP zero-shot поиска (long polling).

- Текстовое сообщение -> поиск изображений по тексту (авто RU->EN перевод).
- Присланное фото -> сохраняется в data/user_photos, добавляется в индекс
  (инкрементально, без пересборки), затем ищутся похожие изображения.

Модель и FAISS-индекс грузятся ОДИН РАЗ при старте (SearchEngine) и живут
резидентно в памяти процесса — см. README, раздел 9.1. Тяжёлый CPU-инференс
выполняется в отдельном потоке (asyncio.to_thread), чтобы не блокировать
event loop бота.

Запуск:
    # локально (нужен .env с TELEGRAM_BOT_TOKEN)
    python -m bot.bot
    # или из папки bot:
    python bot.py

Переменные окружения (см. .env.example):
    TELEGRAM_BOT_TOKEN  — обязателен, токен от @BotFather
    INDEX_DIR           — папка с индексом (по умолчанию "index")
    USER_PHOTOS_DIR     — куда сохранять присланные фото (по умолчанию "data/user_photos")
    TOP_K               — сколько результатов возвращать (по умолчанию 5)
"""

import asyncio
import os
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, InputMediaPhoto, Message

# --- корень проекта в sys.path, чтобы работал импорт "bot.inference" при любом cwd ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.inference import SearchEngine  # noqa: E402

# .env для локального запуска (в Docker переменные приходят через env_file). Необязателен.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
USER_PHOTOS_DIR = Path(os.environ.get("USER_PHOTOS_DIR", "data/user_photos"))
TOP_K = int(os.environ.get("TOP_K", "5"))
# HTTP-прокси для доступа к api.telegram.org (если Telegram режется провайдером).
# На сервере с прямым интернетом не задавать. Пример: http://127.0.0.1:10809
TELEGRAM_PROXY = os.environ.get("TELEGRAM_PROXY", "").strip()

HELP_TEXT = (
    "🔎 <b>CLIP zero-shot поиск по картинкам</b>\n\n"
    "• Пришлите <b>текст</b> (можно по-русски) — найду подходящие изображения.\n"
    "• Пришлите <b>фото</b> — добавлю его в индекс и найду похожие картинки.\n\n"
    "Модель: CLIP ViT-B/32, датасет: MS COCO val2017."
)

# Инициализируется в main() один раз; хендлеры обращаются к готовому объекту.
engine: SearchEngine | None = None
dp = Dispatcher()


def _format_results(results: list[dict]) -> str:
    lines = []
    for rank, r in enumerate(results, start=1):
        lines.append(f"{rank}. id={r['image_id']}  score={r['score']:.3f}")
    return "\n".join(lines)


async def _send_result_images(message: Message, results: list[dict], header: str):
    """Отправляет найденные изображения альбомом; недоступные файлы пропускает."""
    media = []
    for rank, r in enumerate(results, start=1):
        path = r["path"]
        if not Path(path).exists():
            continue
        caption = f"{rank}. id={r['image_id']}  score={r['score']:.3f}"
        media.append(InputMediaPhoto(media=FSInputFile(path), caption=caption))

    if not media:
        await message.answer(f"{header}\n\n(файлы изображений не найдены на диске)")
        return

    await message.answer(header)
    # Telegram допускает до 10 элементов в одном альбоме
    for i in range(0, len(media), 10):
        await message.answer_media_group(media[i:i + 10])


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(HELP_TEXT)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT)


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Фото от пользователя: сохранить -> добавить в индекс -> найти похожие."""
    photo = message.photo[-1]  # берём максимальное разрешение
    USER_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_PHOTOS_DIR / f"{photo.file_unique_id}.jpg"

    await message.bot.download(photo, destination=dest)

    status = await message.answer("⏳ Обрабатываю фото...")

    # инкрементально добавляем в индекс (image_id = file_unique_id, без подписи)
    norm_id, added = await asyncio.to_thread(engine.add_image, str(dest), photo.file_unique_id)

    from PIL import Image
    img = Image.open(dest).convert("RGB")
    similar, captions = await asyncio.to_thread(engine.search_by_image, img, TOP_K)

    added_note = "добавлено в индекс" if added else "уже было в индексе"
    await status.edit_text(f"Фото {added_note} (id={norm_id}). Ищу похожие...")

    await _send_result_images(message, similar, "🖼 <b>Похожие изображения:</b>")

    if captions:
        caps_text = "\n".join(
            f"{i}. {c['caption']}  (score={c['score']:.3f})"
            for i, c in enumerate(captions, start=1)
        )
        await message.answer(f"📝 <b>Наиболее релевантные подписи:</b>\n{caps_text}")


@dp.message(F.text)
async def handle_text(message: Message):
    """Текстовый запрос: поиск изображений (авто RU->EN перевод)."""
    query = message.text.strip()
    if not query:
        return

    status = await message.answer("⏳ Ищу...")
    used_query, results = await asyncio.to_thread(engine.search_by_text, query, TOP_K, True)

    header = f'🔎 <b>Запрос:</b> "{query}"'
    if used_query != query:
        header += f'\n<b>Перевод:</b> "{used_query}"'

    await status.delete()
    await _send_result_images(message, results, header)


async def main():
    global engine

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit(
            "Не задан TELEGRAM_BOT_TOKEN. Укажите его в .env или переменной окружения "
            "(получить токен: @BotFather)."
        )

    print("Инициализация SearchEngine (загрузка модели и индекса)...")
    engine = await asyncio.to_thread(SearchEngine, INDEX_DIR)

    session = AiohttpSession(proxy=TELEGRAM_PROXY) if TELEGRAM_PROXY else None
    if TELEGRAM_PROXY:
        print(f"Подключение к Telegram через прокси: {TELEGRAM_PROXY}")
    bot = Bot(
        token=token,
        session=session,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    print("Бот запущен (long polling). Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
