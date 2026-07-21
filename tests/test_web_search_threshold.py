"""
Порог «только похожее» на уровне API: он должен приходить из настройки и реально
резать выдачу. Базовое app_env отключает порог (-1), поэтому тут переопределяем его
своим значением и сбрасываем кэш настроек.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from web.config import reset_settings


def _jpeg(color=(120, 80, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def filled(client, registered):
    database = client.post("/api/databases", json={"name": "Порог"}).json()
    files = [("files", (f"c{i}.jpg", _jpeg((i * 60 % 256, i * 40 % 256, i * 20 % 256)), "image/jpeg"))
             for i in range(3)]
    client.post(f"/api/databases/{database['id']}/photos", files=files)
    return database


def test_text_threshold_from_settings_empties_weak_results(filled, client, monkeypatch):
    """Недостижимый порог -> пустая выдача. Значит настройка реально доходит до поиска."""
    monkeypatch.setenv("SEARCH_TEXT_MIN_SCORE", "1.5")
    reset_settings()

    body = client.post(
        f"/api/databases/{filled['id']}/search/text",
        json={"query": "что угодно", "top_k": 5, "translate": False},
    ).json()

    assert body["results"] == []


def test_default_env_keeps_search_working(filled, client, monkeypatch):
    """С отключённым порогом (как в app_env) выдача не пуста — регрессия наоборот."""
    body = client.post(
        f"/api/databases/{filled['id']}/search/text",
        json={"query": "что угодно", "top_k": 5, "translate": False},
    ).json()

    assert len(body["results"]) == 3


def test_image_threshold_from_settings(filled, client, monkeypatch):
    monkeypatch.setenv("SEARCH_IMAGE_MIN_SCORE", "1.5")
    reset_settings()

    response = client.post(
        f"/api/databases/{filled['id']}/search/image?top_k=5",
        files={"file": ("q.jpg", _jpeg((10, 20, 30)), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
