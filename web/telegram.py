"""
Проверка данных Telegram Login Widget.

Виджет отдаёт браузеру набор полей и подпись hash. Полям верить нельзя: их прислал
клиент и мог выдумать. Подлинность доказывает только HMAC, посчитанный с ключом,
который знают лишь Telegram и владелец бота, — токеном бота.

Алгоритм описан в https://core.telegram.org/widgets/login (раздел Checking authorization).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

# Данные старше суток не принимаем: перехваченный однажды набор полей иначе
# оставался бы годным ключом от аккаунта навсегда.
MAX_AUTH_AGE_SECONDS = 24 * 60 * 60


class TelegramAuthError(ValueError):
    """Данные виджета не прошли проверку."""


def _data_check_string(payload: dict[str, Any]) -> str:
    """Все поля кроме hash, отсортированные по имени, по одному в строке."""
    return "\n".join(
        f"{key}={payload[key]}" for key in sorted(payload) if key != "hash" and payload[key] is not None
    )


def verify(payload: dict[str, Any], bot_token: str, now: float | None = None) -> dict[str, Any]:
    """
    Проверяет подпись и свежесть. Возвращает очищенные данные пользователя
    или бросает TelegramAuthError.
    """
    if not bot_token:
        raise TelegramAuthError("Вход через Telegram не настроен")

    received_hash = str(payload.get("hash") or "")
    if not received_hash:
        raise TelegramAuthError("В данных нет подписи")

    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    expected = hmac.new(
        secret_key, _data_check_string(payload).encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # сравнение с постоянным временем: обычное == позволяет подбирать подпись побайтно
    if not hmac.compare_digest(expected, received_hash):
        raise TelegramAuthError("Подпись не совпадает")

    try:
        auth_date = int(payload.get("auth_date", 0))
    except (TypeError, ValueError) as e:
        raise TelegramAuthError("Некорректная дата авторизации") from e

    age = (now if now is not None else time.time()) - auth_date
    if age > MAX_AUTH_AGE_SECONDS:
        raise TelegramAuthError("Данные устарели, повторите вход")
    if age < -300:  # запас на расхождение часов
        raise TelegramAuthError("Дата авторизации из будущего")

    telegram_id = str(payload.get("id") or "")
    if not telegram_id:
        raise TelegramAuthError("В данных нет идентификатора пользователя")

    display_name = " ".join(
        part for part in (payload.get("first_name"), payload.get("last_name")) if part
    ).strip()

    return {
        "telegram_id": telegram_id,
        "username": payload.get("username") or None,
        "display_name": display_name or payload.get("username") or f"tg{telegram_id}",
        "photo_url": payload.get("photo_url") or None,
    }
