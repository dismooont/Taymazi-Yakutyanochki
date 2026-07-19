"""
Тесты определения IP клиента за обратным прокси.

Проблема, ради которой всё это написано, проявляется только в продакшене: за nginx
все запросы приходят с одного адреса, и ограничение частоты входа по IP превращается
из защиты в отказ в обслуживании — пятая неудачная попытка любого пользователя
закрывает вход всем остальным.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from web.config import reset_settings
from web.deps import client_ip


def _ip_for(headers: dict[str, str]) -> str:
    """Прогоняет запрос через настоящий стек FastAPI и возвращает вычисленный IP."""
    app = FastAPI()

    @app.get("/whoami")
    def whoami(request: Request) -> dict:
        return {"ip": client_ip(request)}

    with TestClient(app) as probe:
        return probe.get("/whoami", headers=headers).json()["ip"]


def test_direct_connection_uses_peer_address(app_env):
    """Без прокси заголовкам верить нельзя: они пришли прямо от клиента."""
    app_env.setenv("TRUST_PROXY", "0")
    reset_settings()

    assert _ip_for({"X-Real-IP": "203.0.113.7"}) == "testclient"


def test_behind_proxy_uses_real_ip_header(app_env):
    app_env.setenv("TRUST_PROXY", "1")
    reset_settings()

    assert _ip_for({"X-Real-IP": "203.0.113.7"}) == "203.0.113.7"


def test_behind_proxy_takes_last_forwarded_entry(app_env):
    """
    В X-Forwarded-For доверяем только последнему элементу: его дописал прокси,
    а всё, что левее, прислал клиент и мог выдумать.
    """
    app_env.setenv("TRUST_PROXY", "1")
    reset_settings()

    spoofed = {"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 198.51.100.9"}
    assert _ip_for(spoofed) == "198.51.100.9"


def test_different_clients_get_separate_rate_limits(client, app_env):
    """
    Главная проверка: за прокси пользователи не должны делить один счётчик попыток.
    Пять неудач с одного адреса не мешают войти с другого.
    """
    app_env.setenv("TRUST_PROXY", "1")
    reset_settings()

    client.post(
        "/api/auth/register", json={"login": "ivan", "password": "korrektnyy-parol"}
    )
    client.post("/api/auth/logout")

    attacker = {"X-Real-IP": "203.0.113.1"}
    for _ in range(6):
        client.post(
            "/api/auth/login",
            json={"login": "kto-to-drugoy", "password": "nepravilnyy"},
            headers=attacker,
        )

    # честный пользователь с другого адреса входит без помех
    response = client.post(
        "/api/auth/login",
        json={"login": "ivan", "password": "korrektnyy-parol"},
        headers={"X-Real-IP": "198.51.100.2"},
    )
    assert response.status_code == 200
