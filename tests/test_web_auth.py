"""
Тесты авторизации.

Половина проверок здесь — не про «работает ли вход», а про то, что он не выдаёт лишнего:
не даёт перебирать логины, не пускает без заголовка, не оставляет живыми чужие сессии
после смены пароля.
"""

from __future__ import annotations

import pytest

from web import db


# --------------------------------------------------------------------------
# Регистрация
# --------------------------------------------------------------------------

def test_register_creates_user_and_session(client):
    response = client.post(
        "/api/auth/register",
        json={"login": "Ivan", "password": "korrektnyy-parol", "display_name": "Иван"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["login"] == "ivan"  # логин нормализован в нижний регистр
    assert body["display_name"] == "Иван"
    assert body["has_password"] is True
    assert body["has_telegram"] is False

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["id"] == body["id"]


def test_password_is_not_stored_in_plaintext(client, registered):
    user = db.get_user(registered["id"])
    assert user["password_hash"].startswith("$argon2")
    assert "korrektnyy-parol" not in user["password_hash"]


def test_register_rejects_short_password(client):
    response = client.post("/api/auth/register", json={"login": "ivan", "password": "korotk"})
    assert response.status_code == 422
    assert "10 символов" in response.json()["detail"]


def test_register_rejects_common_password(client):
    response = client.post("/api/auth/register", json={"login": "ivan", "password": "password123"})
    assert response.status_code == 422
    assert "распространён" in response.json()["detail"]


@pytest.mark.parametrize("login", ["ab", "иван", "with space", "-starts-with-dash", "a" * 33])
def test_register_rejects_bad_login(client, login):
    response = client.post("/api/auth/register", json={"login": login, "password": "korrektnyy-parol"})
    assert response.status_code == 422


def test_register_rejects_duplicate_login(client, registered):
    response = client.post(
        "/api/auth/register", json={"login": "IVAN", "password": "drugoy-parol-tut"}
    )
    assert response.status_code == 409


def test_registration_can_be_closed(client, app_env):
    """REGISTRATION_OPEN=0 — способ закрыть учебный стенд от посторонних."""
    app_env.setenv("REGISTRATION_OPEN", "0")
    from web.config import reset_settings

    reset_settings()

    response = client.post("/api/auth/register", json={"login": "ivan", "password": "korrektnyy-parol"})
    assert response.status_code == 403


# --------------------------------------------------------------------------
# Вход
# --------------------------------------------------------------------------

def test_login_and_logout(client, registered):
    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401

    response = client.post("/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"})
    assert response.status_code == 200
    assert client.get("/api/me").json()["login"] == "ivan"


def test_login_does_not_leak_existing_logins(client, registered):
    """
    Неверный пароль и несуществующий логин обязаны отвечать одинаково — иначе форма
    входа превращается в справочник зарегистрированных пользователей.
    """
    wrong_password = client.post(
        "/api/auth/login", json={"login": "ivan", "password": "nepravilnyy-parol"}
    )
    unknown_login = client.post(
        "/api/auth/login", json={"login": "nikogo-net", "password": "nepravilnyy-parol"}
    )

    assert wrong_password.status_code == unknown_login.status_code == 401
    assert wrong_password.json()["detail"] == unknown_login.json()["detail"]
    assert wrong_password.json()["detail"] == "Неверный логин или пароль"


def test_login_rate_limited(client, registered):
    for _ in range(5):
        client.post("/api/auth/login", json={"login": "ivan", "password": "nepravilnyy-parol"})

    blocked = client.post("/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"})

    assert blocked.status_code == 429  # даже верный пароль не проходит, пока идёт блокировка


def test_successful_login_resets_attempt_counter(client, registered):
    for _ in range(3):
        client.post("/api/auth/login", json={"login": "ivan", "password": "nepravilnyy-parol"})
    assert client.post(
        "/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"}
    ).status_code == 200

    for _ in range(3):
        client.post("/api/auth/login", json={"login": "ivan", "password": "nepravilnyy-parol"})
    # счётчик обнулился при успешном входе, поэтому три новые опечатки ещё не блокировка
    assert client.post(
        "/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"}
    ).status_code == 200


# --------------------------------------------------------------------------
# Сессии и смена пароля
# --------------------------------------------------------------------------

def test_change_password_closes_other_sessions(client, registered):
    """Пароль меняют, когда подозревают утечку, — чужая сессия должна умереть вместе с ним."""
    from fastapi.testclient import TestClient
    from web.app import create_app

    with TestClient(create_app()) as second_device:
        second_device.headers["X-Requested-With"] = "XMLHttpRequest"
        assert second_device.post(
            "/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"}
        ).status_code == 200
        assert second_device.get("/api/me").status_code == 200

        changed = client.post(
            "/api/me/password",
            json={"old_password": "korrektnyy-parol", "new_password": "novyy-dlinnyy-parol"},
        )
        assert changed.status_code == 200

        assert second_device.get("/api/me").status_code == 401  # второе устройство выкинуло
    assert client.get("/api/me").status_code == 200  # текущее осталось


def test_change_password_requires_old_one(client, registered):
    response = client.post(
        "/api/me/password",
        json={"old_password": "ne-tot-parol", "new_password": "novyy-dlinnyy-parol"},
    )
    assert response.status_code == 403


def test_new_password_works_for_login(client, registered):
    client.post(
        "/api/me/password",
        json={"old_password": "korrektnyy-parol", "new_password": "novyy-dlinnyy-parol"},
    )
    client.post("/api/auth/logout")

    assert client.post(
        "/api/auth/login", json={"login": "ivan", "password": "korrektnyy-parol"}
    ).status_code == 401
    assert client.post(
        "/api/auth/login", json={"login": "ivan", "password": "novyy-dlinnyy-parol"}
    ).status_code == 200


def test_session_token_is_not_stored_as_is(client, registered):
    """В БД лежит только sha256 токена: дамп базы не должен давать вход под чужой сессией."""
    cookie = client.cookies.get("session")
    assert cookie

    with db.connect() as conn:
        stored = [row["token_hash"] for row in conn.execute("SELECT token_hash FROM sessions")]

    assert stored and cookie not in stored


def test_logout_all_closes_current_session_too(client, registered):
    assert client.post("/api/auth/logout-all").json()["closed_sessions"] >= 1
    assert client.get("/api/me").status_code == 401


# --------------------------------------------------------------------------
# Защита запросов
# --------------------------------------------------------------------------

def test_mutating_request_requires_csrf_header(raw_client):
    response = raw_client.post(
        "/api/auth/register", json={"login": "ivan", "password": "korrektnyy-parol"}
    )
    assert response.status_code == 403
    assert "X-Requested-With" in response.json()["detail"]


def test_safe_request_does_not_require_header(raw_client):
    assert raw_client.get("/api/health").json() == {"status": "ok"}


def test_me_requires_authentication(client):
    assert client.get("/api/me").status_code == 401


def test_expired_session_is_rejected(client, registered):
    """Протухшая сессия не должна пускать, даже если cookie ещё лежит в браузере."""
    with db.connect() as conn:
        conn.execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00+00:00'")

    assert client.get("/api/me").status_code == 401
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
