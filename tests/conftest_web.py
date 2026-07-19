"""
Фикстуры веб-тестов вынесены отдельным модулем и подключаются из tests/conftest.py,
чтобы тесты ядра (tests/test_store.py) не тянули за собой FastAPI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.config import reset_settings
from web.routers.search import search_limiter
from web.security import login_limiter, register_limiter
from web.stores import store_cache


@pytest.fixture
def app_env(tmp_path, monkeypatch, holder):
    """Изолированное окружение: своя папка данных и своя SQLite на каждый тест."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTRATION_OPEN", "1")
    monkeypatch.setenv("PUBLIC_URL", "http://localhost:5173")
    monkeypatch.setenv("MIN_PASSWORD_LENGTH", "10")
    # Демо-база подключается, только если указанный индекс существует. В тестах путь
    # заведомо пустой: иначе настоящий индекс COCO из index/ подмешивался бы в списки
    # баз и ломал проверки, а тесты зависели бы от машины, на которой запущены.
    monkeypatch.setenv("DEMO_INDEX_DIR", str(tmp_path / "no-demo"))
    # Бот живёт в том же процессе. Без этого тесты, задающие токен, полезли бы
    # в настоящий Telegram.
    monkeypatch.setenv("TELEGRAM_BOT_ENABLED", "0")
    reset_settings()
    login_limiter.clear()
    register_limiter.clear()
    search_limiter.clear()
    store_cache.clear()
    yield monkeypatch
    reset_settings()
    store_cache.clear()


@pytest.fixture
def raw_client(app_env):
    """Клиент без заголовка X-Requested-With — им проверяется защита от CSRF."""
    from web.app import create_app

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def client(raw_client):
    """Обычный клиент: ведёт себя как фронтенд, всегда шлёт X-Requested-With."""
    raw_client.headers["X-Requested-With"] = "XMLHttpRequest"
    return raw_client


@pytest.fixture
def registered(client):
    """Зарегистрированный и вошедший пользователь."""
    response = client.post(
        "/api/auth/register",
        json={"login": "ivan", "password": "korrektnyy-parol", "display_name": "Иван"},
    )
    assert response.status_code == 201, response.text
    return response.json()
