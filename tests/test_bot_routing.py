"""
Тесты маршрутизации сообщений в боте.

Aiogram отдаёт сообщение первому подходящему обработчику, поэтому порядок
регистрации — это поведение, а не оформление. На живом боте это уже сломалось:
обработчик текста стоял раньше /demo, и «/demo красный автобус» уходил в поиск
по базе чата вместо демо-базы.
"""

from __future__ import annotations

import datetime

import pytest
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Chat, Message, PhotoSize, User

from bot.bot import build_dispatcher
from bot.client import SearchApi

# Токен нужен фильтру Command: он спрашивает у бота его имя, чтобы понимать
# обращения вида /demo@my_bot. Запросов в сеть при проверке фильтров нет.
FAKE_TOKEN = "123456:AAHfake-token-for-filter-checks-only"


def _message(text: str, chat_type: str = "private") -> Message:
    return Message(
        message_id=1,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=Chat(id=42, type=chat_type),
        from_user=User(id=7, is_bot=False, first_name="Иван"),
        text=text,
    )


@pytest.fixture
def handlers():
    dispatcher = build_dispatcher(SearchApi("http://example", "token"))
    return dispatcher.message.handlers


@pytest.fixture
def bot():
    return Bot(token=FAKE_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))


async def _first_match(handlers, bot: Bot, message: Message) -> str:
    """Имя обработчика, которому aiogram отдаст сообщение, — его же механизмом."""
    for handler in handlers:
        passed, _ = await handler.check(message, bot=bot)
        if passed:
            return handler.callback.__name__
    return "нет обработчика"


@pytest.mark.asyncio
@pytest.mark.parametrize("text, expected", [
    ("/demo красный автобус", "on_demo_cmd"),
    ("/demo", "on_demo_cmd"),
    ("/find собака", "on_find_cmd"),
    ("/start", "on_start"),
    ("/stats", "on_stats"),
    ("/help", "on_help"),
    ("рыжий кот на подоконнике", "on_private_text"),
    ("/чтототакое", "on_unknown_command"),
])
async def test_private_routing(handlers, bot, text, expected):
    assert await _first_match(handlers, bot, _message(text)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("text, expected", [
    ("🔎 Найти", "on_find_btn"),
    ("🎞 Демо", "on_demo_btn"),
    ("📊 Статистика", "on_stats"),
    ("📦 Архив", "on_export"),
    ("❓ Помощь", "on_help_btn"),
])
async def test_menu_buttons_route_to_actions(handlers, bot, text, expected):
    """
    Кнопки нижнего меню — это обычный текст. Их обработчики обязаны стоять раньше
    свободного текста, иначе «🔎 Найти» ушло бы в поиск как запрос.
    """
    assert await _first_match(handlers, bot, _message(text)) == expected


@pytest.mark.asyncio
async def test_group_text_is_ignored(handlers, bot):
    """В группе бот не лезет в каждое сообщение — только команды."""
    found = await _first_match(handlers, bot, _message("просто болтовня", "supergroup"))
    assert found == "нет обработчика"


@pytest.mark.asyncio
async def test_group_demo_works(handlers, bot):
    assert await _first_match(handlers, bot, _message("/demo кот", "supergroup")) == "on_demo_cmd"


# --------------------------------------------------------------------------
# Каналы: пост канала — отдельный тип апдейта (channel_post), не message
# --------------------------------------------------------------------------

def _channel_photo() -> Message:
    return Message(
        message_id=1,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=Chat(id=-1001, type="channel", title="Канал"),
        photo=[PhotoSize(file_id="f", file_unique_id="u", width=90, height=90, file_size=1)],
    )


@pytest.mark.asyncio
async def test_channel_photo_routes_to_channel_handler(bot):
    """Фото-пост канала должен попасть в обработчик channel_post, а не потеряться."""
    dispatcher = build_dispatcher(SearchApi("http://example", "token"))
    found = await _first_match(dispatcher.channel_post.handlers, bot, _channel_photo())
    assert found == "on_channel_photo"


def test_channel_and_membership_handlers_registered():
    """Каналы держатся на двух отдельных наблюдателях — оба должны быть на месте."""
    dispatcher = build_dispatcher(SearchApi("http://example", "token"))
    assert len(dispatcher.channel_post.handlers) == 1
    assert len(dispatcher.my_chat_member.handlers) == 1
