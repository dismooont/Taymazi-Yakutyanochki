"""
Тесты поиска и отдачи файлов.

Качество ранжирования CLIP здесь не проверяется — тесты идут на подменённой модели
(см. tests/conftest.py), за качество отвечает scripts/eval_recall.py. Проверяется другое:
что выдача указывает на правильные фотографии, что файлы отдаются только владельцу
и что по photo_id из запроса нельзя выбраться за пределы своей базы.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


def _jpeg(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


COLORS = [(10, 20, 30), (200, 30, 40), (30, 200, 40), (40, 30, 200)]


@pytest.fixture
def filled(client, registered):
    """База с четырьмя разными снимками; возвращает (база, [(цвет, photo_id)])."""
    database = client.post("/api/databases", json={"name": "Поиск"}).json()
    files = [("files", (f"c{i}.jpg", _jpeg(color), "image/jpeg")) for i, color in enumerate(COLORS)]
    # по три за раз, чтобы обработка шла синхронно и тест не ждал очередь
    client.post(f"/api/databases/{database['id']}/photos", files=files[:3])
    client.post(f"/api/databases/{database['id']}/photos", files=files[3:])

    items = client.get(f"/api/databases/{database['id']}/photos").json()["items"]
    return database, items


# --------------------------------------------------------------------------
# Поиск по тексту
# --------------------------------------------------------------------------

def test_search_text_returns_hits(client, filled):
    database, _ = filled

    response = client.post(
        f"/api/databases/{database['id']}/search/text",
        json={"query": "orange cat", "top_k": 3, "translate": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_query"] == "orange cat"
    assert len(body["results"]) == 3
    scores = [hit["score"] for hit in body["results"]]
    assert scores == sorted(scores, reverse=True)  # выдача отсортирована по убыванию


def test_search_result_urls_are_usable(client, filled):
    """Ссылки из выдачи фронт вставляет в <img> как есть — они обязаны работать."""
    database, _ = filled

    hit = client.post(
        f"/api/databases/{database['id']}/search/text",
        json={"query": "anything", "translate": False},
    ).json()["results"][0]

    thumb = client.get(hit["thumb_url"])
    original = client.get(hit["file_url"])

    assert thumb.status_code == 200
    assert original.status_code == 200
    assert len(original.content) > 0


def test_search_in_empty_database(client, registered):
    database = client.post("/api/databases", json={"name": "Пустая"}).json()

    response = client.post(
        f"/api/databases/{database['id']}/search/text",
        json={"query": "что угодно", "translate": False},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_top_k_larger_than_database(client, filled):
    database, items = filled

    response = client.post(
        f"/api/databases/{database['id']}/search/text",
        json={"query": "test", "top_k": 50, "translate": False},
    )

    assert len(response.json()["results"]) == len(items)


def test_search_is_rate_limited(client, filled):
    database, _ = filled
    payload = {"query": "spam", "translate": False}

    codes = [
        client.post(f"/api/databases/{database['id']}/search/text", json=payload).status_code
        for _ in range(35)
    ]

    assert 429 in codes
    assert codes.count(200) == 30


# --------------------------------------------------------------------------
# Поиск по картинке
# --------------------------------------------------------------------------

def test_search_by_image_finds_the_same_photo(client, filled):
    """Снимок, уже лежащий в базе, обязан найти сам себя первым результатом."""
    database, _ = filled
    query = _jpeg(COLORS[2])

    response = client.post(
        f"/api/databases/{database['id']}/search/image",
        files={"file": ("query.jpg", query, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert body["captions"] == []  # у пользовательской базы подписей нет

    same = client.get(body["results"][0]["file_url"])
    assert same.content == query


def test_search_by_image_rejects_non_image(client, filled):
    database, _ = filled

    response = client.post(
        f"/api/databases/{database['id']}/search/image",
        files={"file": ("notes.txt", b"hello there", "text/plain")},
    )

    assert response.status_code == 422
    assert "не является изображением" in response.json()["detail"]


def test_query_image_is_not_stored(client, filled):
    """Картинка-образец нужна на один прогон энкодера и не должна оседать в базе."""
    database, items = filled

    client.post(
        f"/api/databases/{database['id']}/search/image",
        files={"file": ("query.jpg", _jpeg((7, 7, 7)), "image/jpeg")},
    )

    assert client.get(f"/api/databases/{database['id']}/photos").json()["total"] == len(items)


# --------------------------------------------------------------------------
# Отдача файлов
# --------------------------------------------------------------------------

def test_thumb_and_file_served(client, filled):
    database, items = filled
    photo_id = items[0]["photo_id"]

    thumb = client.get(f"/api/databases/{database['id']}/photos/{photo_id}/thumb")
    original = client.get(f"/api/databases/{database['id']}/photos/{photo_id}/file")

    assert thumb.status_code == 200
    assert len(thumb.content) > 0
    assert original.status_code == 200
    assert len(original.content) > len(thumb.content)  # превью легче оригинала


def test_files_are_cached_immutably(client, filled):
    """photo_id — хеш содержимого, поэтому файл по адресу не меняется никогда."""
    database, items = filled

    response = client.get(f"/api/databases/{database['id']}/photos/{items[0]['photo_id']}/file")

    cache_control = response.headers["cache-control"]
    assert "immutable" in cache_control
    assert "private" in cache_control  # не должно оседать в общих прокси


def test_unknown_photo_is_404(client, filled):
    database, _ = filled
    assert client.get(f"/api/databases/{database['id']}/photos/deadbeef/file").status_code == 404


@pytest.mark.parametrize("evil_id", [
    "..%2F..%2F..%2Fapp.db",
    "....//....//app.db",
    "%2e%2e%2f%2e%2e%2fapp.db",
])
def test_photo_id_cannot_escape_database_folder(client, filled, evil_id):
    """photo_id приходит от пользователя — по нему нельзя добраться до чужих файлов."""
    database, _ = filled

    response = client.get(f"/api/databases/{database['id']}/photos/{evil_id}/file")

    assert response.status_code == 404
    assert b"sqlite" not in response.content.lower()


def test_foreign_photos_are_not_served(client, filled):
    """Главная причина, по которой файлы отдаются через API, а не статикой из data/."""
    from tests.test_web_databases import _second_user

    database, items = filled
    photo_id = items[0]["photo_id"]
    attacker = _second_user("mallory4")
    try:
        assert attacker.get(
            f"/api/databases/{database['id']}/photos/{photo_id}/file"
        ).status_code == 404
        assert attacker.post(
            f"/api/databases/{database['id']}/search/text",
            json={"query": "x", "translate": False},
        ).status_code == 404
    finally:
        attacker.__exit__(None, None, None)


def test_photos_require_authentication(raw_client, filled):
    database, items = filled
    raw_client.cookies.clear()

    response = raw_client.get(f"/api/databases/{database['id']}/photos/{items[0]['photo_id']}/file")

    assert response.status_code == 401
