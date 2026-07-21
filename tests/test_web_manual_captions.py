"""
Тесты ручных подписей к снимкам (пользовательская фича №4).

Сама механика хранения проверена в test_store_captions.py. Здесь — обвязка API:
что подпись сохраняется и доезжает до списка фото и до выдачи поиска, что демо-базу
править нельзя, что пустая строка снимает подпись, и что при установленном энкодере
подпись сразу становится искомой (векторизуется), а без него — хранится как текст.

Текстовая модель не грузится: где нужно, энкодер подменяется заглушкой.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from web import db, stores


def _jpeg(seed: int) -> bytes:
    buffer = io.BytesIO()
    color = (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def photo(client, registered):
    """База с одним снимком; возвращает (id базы, photo_id)."""
    database = client.post("/api/databases", json={"name": "Подписи"}).json()
    client.post(f"/api/databases/{database['id']}/photos",
                files=[("files", ("a.jpg", _jpeg(1), "image/jpeg"))])
    items = client.get(f"/api/databases/{database['id']}/photos").json()["items"]
    return database["id"], items[0]["photo_id"]


def _set(client, db_id, photo_id, caption):
    return client.put(f"/api/databases/{db_id}/photos/{photo_id}/caption",
                      json={"caption": caption})


# --------------------------------------------------------------------------
# Сохранение и отображение
# --------------------------------------------------------------------------

def test_caption_is_saved_and_listed(client, photo):
    db_id, photo_id = photo

    response = _set(client, db_id, photo_id, "рыжий кот на подоконнике")
    assert response.status_code == 200
    assert response.json()["caption"] == "рыжий кот на подоконнике"

    item = client.get(f"/api/databases/{db_id}/photos").json()["items"][0]
    assert item["caption"] == "рыжий кот на подоконнике"


def test_caption_is_trimmed(client, photo):
    db_id, photo_id = photo
    assert _set(client, db_id, photo_id, "  кот  ").json()["caption"] == "кот"


def test_caption_can_be_edited(client, photo):
    db_id, photo_id = photo
    _set(client, db_id, photo_id, "первая")
    assert _set(client, db_id, photo_id, "вторая").json()["caption"] == "вторая"
    assert client.get(f"/api/databases/{db_id}/photos").json()["items"][0]["caption"] == "вторая"


def test_empty_caption_clears_it(client, photo):
    db_id, photo_id = photo
    _set(client, db_id, photo_id, "убрать")

    assert _set(client, db_id, photo_id, "").json()["caption"] == ""
    assert client.get(f"/api/databases/{db_id}/photos").json()["items"][0]["caption"] == ""


def test_caption_updates_coverage(client, photo):
    """Покрытие подписями видно в шапке базы — оно должно сдвинуться."""
    db_id, photo_id = photo
    assert client.get(f"/api/databases/{db_id}").json()["captions_count"] == 0

    _set(client, db_id, photo_id, "есть подпись")

    assert client.get(f"/api/databases/{db_id}").json()["captions_count"] == 1


# --------------------------------------------------------------------------
# Границы и права
# --------------------------------------------------------------------------

def test_unknown_photo_is_404(client, photo):
    db_id, _ = photo
    assert _set(client, db_id, "нет-такого-id", "текст").status_code == 404


def test_too_long_caption_rejected(client, photo):
    db_id, photo_id = photo
    assert _set(client, db_id, photo_id, "я" * 501).status_code == 422


def test_foreign_database_is_not_accessible(client, photo):
    """Чужая база — 404, как и для остальных операций над ней."""
    db_id, photo_id = photo
    # второй пользователь
    client.post("/api/auth/logout")
    client.post("/api/auth/register",
                json={"login": "petya", "password": "korrektnyy-parol"})
    assert _set(client, db_id, photo_id, "чужое").status_code == 404


# --------------------------------------------------------------------------
# Связь с поиском по подписям
# --------------------------------------------------------------------------

class _FakeEncoder:
    model_name = "fake"

    @staticmethod
    def encode_one(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=8).astype("float32")
        return v / np.linalg.norm(v)


def test_caption_is_indexed_when_encoder_available(client, photo, monkeypatch):
    """С установленным энкодером ручная подпись сразу векторизуется."""
    db_id, photo_id = photo
    monkeypatch.setattr(stores, "caption_encoder_available", lambda: True)
    monkeypatch.setattr(stores.CaptionEncoder, "get", classmethod(lambda cls, name: _FakeEncoder()))

    body = _set(client, db_id, photo_id, "красный автобус").json()
    assert body["indexed"] is True

    store = stores.store_for(db.get_database(db_id))
    assert store.caption_index_model()  # вектор действительно записан


def test_caption_is_text_only_without_encoder(client, photo, monkeypatch):
    """Без библиотеки подпись хранится как текст — не падаем и не индексируем."""
    db_id, photo_id = photo
    monkeypatch.setattr(stores, "caption_encoder_available", lambda: False)

    body = _set(client, db_id, photo_id, "кот").json()

    assert body["indexed"] is False
    assert body["caption"] == "кот"  # но сама подпись сохранена
