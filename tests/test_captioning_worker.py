"""
Тесты фоновой разметки (фаза C4).

BLIP и SBERT здесь не грузятся — оба подменяются. Проверяется не качество подписей
(это C0–C3), а планировщик: когда он работает, когда отступает и почему. Каждое из
трёх правил замысла — простой, вытеснение задачей, гарантия прогресса — по
отдельности решает, портит система отклик или нет, поэтому тестируется каждое.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from web import captioning, db
from web.captioning import CaptionWorker, activity
from web.config import reset_settings


class FakeCaptioner:
    """Вместо BLIP — фраза по имени файла. Считает вызовы, весов не грузит."""

    model_name = "fake-blip"

    def __init__(self):
        self.calls = 0

    def caption_images(self, paths, batch_size: int = 8):
        self.calls += 1
        return [f"подпись {i}" for i, _ in enumerate(paths)]


class FakeEncoder:
    model_name = "fake-enc"

    @staticmethod
    def encode(texts):
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 8)).astype("float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def wired(client, registered, monkeypatch):
    """Приложение с базой из 12 снимков и подменёнными моделями."""
    monkeypatch.setattr(captioning.Captioner, "get",
                        classmethod(lambda cls, name, num_threads=None: FakeCaptioner()))
    monkeypatch.setattr(captioning.CaptionEncoder, "get",
                        classmethod(lambda cls, name: FakeEncoder()))
    monkeypatch.setattr(captioning, "caption_encoder_available", lambda: True)

    database = client.post("/api/databases", json={"name": "Разметка"}).json()
    for start in range(0, 12, 3):
        files = [("files", (f"p{i}.jpg", _jpeg(i), "image/jpeg")) for i in range(start, start + 3)]
        client.post(f"/api/databases/{database['id']}/photos", files=files)
    return database


def _jpeg(seed: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (seed * 9 % 256, seed * 5 % 256, seed * 3 % 256)).save(buf, "JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Когда работать, когда отступать
# --------------------------------------------------------------------------

def test_works_after_idle(wired, monkeypatch):
    monkeypatch.setenv("CAPTION_IDLE_SECONDS", "30")
    reset_settings()
    worker = CaptionWorker()

    activity.touch()
    assert worker.may_work() is False  # только что был запрос

    # притворяемся, что запросов не было полминуты
    monkeypatch.setattr(activity, "idle_seconds", lambda: 31.0)
    assert worker.may_work() is True


def test_active_job_blocks_captioning(wired, monkeypatch):
    """Пользовательская задача важнее: её человек ждёт с прогрессом на экране."""
    monkeypatch.setattr(activity, "idle_seconds", lambda: 999.0)
    worker = CaptionWorker()
    assert worker.may_work() is True

    monkeypatch.setattr(db, "has_any_active_job", lambda: True)
    assert worker.may_work() is False


def test_forced_run_prevents_starvation(wired, monkeypatch):
    """
    Активный пользователь не должен навсегда лишить свою базу подписей: раз в
    caption_force_after разметка идёт даже без простоя.
    """
    monkeypatch.setenv("CAPTION_IDLE_SECONDS", "30")
    monkeypatch.setenv("CAPTION_FORCE_AFTER", "600")
    reset_settings()
    worker = CaptionWorker()

    monkeypatch.setattr(activity, "idle_seconds", lambda: 5.0)  # всё время занят
    assert worker.may_work() is False

    worker._last_run = time.monotonic() - 601  # десять минут не работали
    assert worker.may_work() is True


# --------------------------------------------------------------------------
# Собственно разметка
# --------------------------------------------------------------------------

def test_slice_captions_and_encodes(wired):
    database = wired
    worker = CaptionWorker()

    written = worker.run_slice()

    assert written > 0
    from web.stores import store_for

    store = store_for(db.get_database(database["id"]))
    covered, total = store.captions_coverage()
    assert covered == written
    assert total == 12


def test_full_pass_covers_and_enables_fusion(wired):
    database = wired
    worker = CaptionWorker()

    for _ in range(10):  # 12 снимков по 8 за отрезок — двух хватает с запасом
        if worker.run_slice() == 0:
            break

    from web.stores import store_for

    store = store_for(db.get_database(database["id"]))
    assert store.captions_coverage() == (12, 12)
    assert store.fusion_ready() is True

    row = db.get_database(database["id"])
    assert row["captions_count"] == 12  # покрытие доехало до SQLite


def test_coverage_reaches_the_api(wired, client):
    database = wired
    worker = CaptionWorker()
    worker.run_slice()

    body = client.get(f"/api/databases/{database['id']}").json()
    assert body["captions_count"] > 0
    assert body["captions_count"] <= body["photos_count"]


def test_nothing_to_do_is_harmless(wired):
    worker = CaptionWorker()
    while worker.run_slice() > 0:
        pass
    assert worker.run_slice() == 0  # всё размечено — тихий ноль, не ошибка


def test_demo_and_empty_bases_are_skipped(wired, client, monkeypatch):
    """Размечать нечего в пустой базе и нельзя в демо (она только для чтения)."""
    client.post("/api/databases", json={"name": "Пустая"})  # 0 снимков

    captionable = db.list_captionable_databases()
    names = {d["name"] for d in captionable}
    assert "Разметка" in names
    assert "Пустая" not in names
    assert all(d["kind"] != "demo" for d in captionable)
