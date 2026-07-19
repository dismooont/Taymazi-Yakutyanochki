"""
Тесты входа через Telegram Login Widget.

Виджет присылает поля в открытом виде — им нельзя верить ни на грамм. Подлинность
доказывает только HMAC с ключом из токена бота, поэтому большая часть тестов здесь
про то, что подделанные и просроченные данные не проходят.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from web import db, telegram
from web.config import reset_settings

BOT_TOKEN = "123456:TEST-TOKEN-FOR-TESTS"
BOT_NAME = "test_search_bot"


def sign(payload: dict, token: str = BOT_TOKEN) -> dict:
    """Подписывает набор полей так же, как это делает Telegram."""
    check_string = "\n".join(f"{k}={payload[k]}" for k in sorted(payload) if k != "hash")
    secret = hashlib.sha256(token.encode()).digest()
    payload = dict(payload)
    payload["hash"] = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    return payload


def widget_payload(telegram_id: str = "777", **extra) -> dict:
    return sign({
        "id": telegram_id,
        "first_name": "Иван",
        "username": "ivan_tg",
        "auth_date": int(time.time()),
        **extra,
    })


@pytest.fixture
def telegram_on(app_env):
    app_env.setenv("TELEGRAM_AUTH_ENABLED", "1")
    app_env.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    app_env.setenv("TELEGRAM_BOT_USERNAME", BOT_NAME)
    reset_settings()
    return app_env


# --------------------------------------------------------------------------
# Проверка подписи
# --------------------------------------------------------------------------

def test_valid_payload_accepted():
    data = telegram.verify(widget_payload(), BOT_TOKEN)

    assert data["telegram_id"] == "777"
    assert data["display_name"] == "Иван"


def test_tampered_field_rejected():
    """Поля и подпись связаны: поменяли id — подпись перестала сходиться."""
    payload = widget_payload("777")
    payload["id"] = "999"

    with pytest.raises(telegram.TelegramAuthError, match="Подпись"):
        telegram.verify(payload, BOT_TOKEN)


def test_payload_signed_with_other_token_rejected():
    """Подпись чужим токеном не годится: ключ знают только Telegram и владелец бота."""
    payload = sign({"id": "777", "auth_date": int(time.time())}, token="999999:CHUZHOY-TOKEN")

    with pytest.raises(telegram.TelegramAuthError, match="Подпись"):
        telegram.verify(payload, BOT_TOKEN)


def test_stale_payload_rejected():
    """Перехваченный набор полей не должен работать вечно."""
    old = widget_payload(auth_date=int(time.time()) - 48 * 3600)

    with pytest.raises(telegram.TelegramAuthError, match="устарели"):
        telegram.verify(old, BOT_TOKEN)


def test_payload_from_future_rejected():
    future = widget_payload(auth_date=int(time.time()) + 3600)

    with pytest.raises(telegram.TelegramAuthError, match="будущего"):
        telegram.verify(future, BOT_TOKEN)


def test_missing_hash_rejected():
    with pytest.raises(telegram.TelegramAuthError, match="подписи"):
        telegram.verify({"id": "777", "auth_date": int(time.time())}, BOT_TOKEN)


# --------------------------------------------------------------------------
# Вход через API
# --------------------------------------------------------------------------

def test_first_telegram_login_creates_account(client, telegram_on):
    response = client.post("/api/auth/telegram", json=widget_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Иван"
    assert body["has_telegram"] is True
    assert body["has_password"] is False  # пароля у такого аккаунта ещё нет
    assert client.get("/api/me").json()["id"] == body["id"]


def test_second_login_reuses_same_account(client, telegram_on):
    first = client.post("/api/auth/telegram", json=widget_payload()).json()
    client.post("/api/auth/logout")

    second = client.post("/api/auth/telegram", json=widget_payload()).json()

    assert second["id"] == first["id"]  # второй аккаунт не заводится


def test_forged_login_rejected(client, telegram_on):
    payload = widget_payload()
    payload["hash"] = "0" * 64

    assert client.post("/api/auth/telegram", json=payload).status_code == 401


def test_telegram_disabled_by_default(client, app_env):
    """Пока вход не настроен, ручки нет вовсе — кнопка без работающего входа хуже её отсутствия."""
    assert client.post("/api/auth/telegram", json=widget_payload()).status_code == 404


def test_config_advertises_telegram(client, telegram_on):
    config = client.get("/api/config").json()

    assert config["telegram_auth"] is True
    assert config["telegram_bot"] == BOT_NAME


def test_config_hides_telegram_when_token_missing(client, app_env):
    """Включённый флаг без токена — полувключённое состояние, его быть не должно."""
    app_env.setenv("TELEGRAM_AUTH_ENABLED", "1")
    app_env.setenv("TELEGRAM_BOT_TOKEN", "")
    reset_settings()

    config = client.get("/api/config").json()

    assert config["telegram_auth"] is False
    assert config["telegram_bot"] is None


# --------------------------------------------------------------------------
# Привязка к существующему аккаунту
# --------------------------------------------------------------------------

def test_link_to_password_account(client, registered, telegram_on):
    linked = client.post("/api/auth/telegram", json=widget_payload())

    assert linked.status_code == 200
    body = linked.json()
    assert body["id"] == registered["id"]  # аккаунт тот же, не новый
    assert body["has_password"] is True
    assert body["has_telegram"] is True


def test_linked_telegram_logs_into_same_account(client, registered, telegram_on):
    client.post("/api/auth/telegram", json=widget_payload())
    client.post("/api/auth/logout")

    entered = client.post("/api/auth/telegram", json=widget_payload()).json()

    assert entered["id"] == registered["id"]
    assert entered["login"] == "ivan"


def test_cannot_steal_linked_telegram(client, registered, telegram_on):
    """Один Telegram — один аккаунт: иначе чужой аккаунт уводится привязкой."""
    client.post("/api/auth/telegram", json=widget_payload())
    client.post("/api/auth/logout")
    client.post("/api/auth/register", json={"login": "petr", "password": "korrektnyy-parol"})

    response = client.post("/api/auth/telegram", json=widget_payload())

    assert response.status_code == 409
    assert "другому аккаунту" in response.json()["detail"]


def test_unlink_requires_password(client, telegram_on):
    """Отвязка у аккаунта без пароля оставила бы его без единого способа входа."""
    client.post("/api/auth/telegram", json=widget_payload())

    response = client.delete("/api/me/identities/telegram")

    assert response.status_code == 409
    assert "задайте пароль" in response.json()["detail"]


def test_unlink_after_setting_password(client, telegram_on):
    client.post("/api/auth/telegram", json=widget_payload())
    # у аккаунта из Telegram пароля нет, поэтому старый не спрашивается
    client.post("/api/me/password", json={"old_password": "", "new_password": "novyy-dlinnyy-parol"})

    response = client.delete("/api/me/identities/telegram")

    assert response.status_code == 200
    assert response.json()["has_telegram"] is False
    assert db.get_user_by_identity("telegram", "777") is None
