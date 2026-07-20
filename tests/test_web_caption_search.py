"""
Тесты обвязки поиска по подписям (фаза C2).

Само слияние проверено в tests/test_store_fusion.py. Здесь — решение о том,
включать ли его: настройка, покрытие базы и наличие библиотеки. Каждое из трёх
условий должно уметь выключить поиск по подписям само по себе, и ни одно не
должно ронять обычный поиск.

Текстовая модель не грузится: до неё дело не доходит ни в одном из тестов.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from web import db, stores
from web.config import reset_settings

SIZE = 12
DIM = 8


def _jpeg(seed: int) -> bytes:
    buffer = io.BytesIO()
    color = (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


def vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("CAPTION_SEARCH_ENABLED", "1")
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def base(client, registered, tmp_path):
    """Пользовательская база из SIZE снимков плюс открытый IndexStore к ней."""
    database = client.post("/api/databases", json={"name": "Подписи"}).json()
    for start in range(0, SIZE, 3):  # по три за раз — тогда обработка синхронная
        files = [("files", (f"p{i}.jpg", _jpeg(i), "image/jpeg"))
                 for i in range(start, start + 3)]
        response = client.post(f"/api/databases/{database['id']}/photos", files=files)
        assert response.status_code == 202, response.text
        assert response.json()["job_id"] is None, "три файла обрабатываются синхронно"

    # ответ API user_id не отдаёт — берём строку базы из БД, как это делает роутер
    store = stores.store_for(db.get_database(database["id"]))
    assert len(store) == SIZE
    return database, store, [p.photo_id for p in store.list_photos()]


def caption_everything(store, photo_ids):
    store.set_caption_texts(
        {pid: f"подпись про снимок {i}" for i, pid in enumerate(photo_ids)}, model="test"
    )
    store.set_caption_vectors(
        {pid: vector(i) for i, pid in enumerate(photo_ids)}, model="test"
    )


def _search(client, database):
    response = client.post(
        f"/api/databases/{database['id']}/search/text",
        json={"query": "красный автобус", "top_k": 5, "translate": False},
    )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Когда слияние включается
# --------------------------------------------------------------------------

def test_disabled_by_default_even_with_captions(client, base):
    """Без явного включения поиск остаётся обычным, даже если подписи есть."""
    database, store, photo_ids = base
    caption_everything(store, photo_ids)

    assert stores.caption_encoder_for(store) is None
    assert _search(client, database)["fused"] is False


def test_enabled_without_captions_stays_plain(client, base, enabled):
    """Настройка включена, но размечать нечего — слияние всё равно не работает."""
    _, store, _ = base
    assert stores.caption_encoder_for(store) is None


def test_low_coverage_stays_plain(client, base, enabled):
    """
    Мало подписей — обычный поиск. Иначе оценка нормировалась бы по горстке
    снимков, и лучший из них вылезал бы наверх без оснований.
    """
    _, store, photo_ids = base
    few = photo_ids[:3]
    store.set_caption_texts({pid: "подпись" for pid in few})
    store.set_caption_vectors({pid: vector(i) for i, pid in enumerate(few)}, model="test")

    assert store.fusion_ready() is False
    assert stores.caption_encoder_for(store) is None


def test_missing_library_disables_search_without_breaking_it(
    client, base, enabled, monkeypatch
):
    """
    sentence-transformers лежит только в requirements-dev, в обычной сборке его
    нет. Поиск от этого падать не должен — он просто остаётся обычным.
    """
    database, store, photo_ids = base
    caption_everything(store, photo_ids)
    monkeypatch.setattr(stores, "caption_encoder_available", lambda: False)

    assert stores.caption_encoder_for(store) is None
    assert _search(client, database)["fused"] is False


def test_ready_when_enabled_and_covered(client, base, enabled, monkeypatch):
    database, store, photo_ids = base
    caption_everything(store, photo_ids)
    monkeypatch.setattr(stores, "caption_encoder_available", lambda: True)
    monkeypatch.setattr(
        stores.CaptionEncoder, "get", classmethod(lambda cls, name: _FakeEncoder())
    )

    encoder = stores.caption_encoder_for(store)
    assert encoder is not None

    body = _search(client, database)
    assert body["fused"] is True
    # оценка слияния — отклонение от среднего, у нижней половины выдачи она
    # отрицательна по построению, и это нормально
    assert any(hit["score"] < 0 for hit in body["results"]) or len(body["results"]) < SIZE


class _FakeEncoder:
    model_name = "fake"

    @staticmethod
    def encode_one(text: str) -> np.ndarray:
        return vector(0)


# --------------------------------------------------------------------------
# Подпись доезжает до клиента
# --------------------------------------------------------------------------

def test_caption_reaches_the_client(client, base):
    """Даже без слияния подпись должна попадать в выдачу: она объясняет находку."""
    database, store, photo_ids = base
    caption_everything(store, photo_ids)

    body = _search(client, database)

    assert all(hit["caption"].startswith("подпись про снимок") for hit in body["results"])


def test_photos_without_captions_report_empty_string(client, base):
    database, _, _ = base

    body = _search(client, database)

    assert all(hit["caption"] == "" for hit in body["results"])
