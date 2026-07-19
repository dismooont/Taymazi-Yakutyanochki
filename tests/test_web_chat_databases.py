"""
Тесты баз, привязанных к Telegram-чатам, и демо-базы.

Главное здесь — границы доступа. База чата видна участникам, но не посторонним;
демо-база видна всем, но менять её нельзя никому.
"""

from __future__ import annotations

import io
import json
import zipfile

import faiss
import numpy as np
import pytest
from PIL import Image

from web import db
from web.config import reset_settings


def _jpeg(color=(90, 140, 190)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def chat_database(client, registered):
    """База чата, где зарегистрированный пользователь — участник."""
    database = db.create_database(
        registered["id"], "Чат путешественников", kind="chat", telegram_chat_id="-100500"
    )
    from web.stores import create_store

    create_store(registered["id"], database["id"])
    db.remember_chat_member("-100500", registered["id"])
    return database


def _other_user(login: str):
    from tests.test_web_databases import _second_user

    return _second_user(login)


# --------------------------------------------------------------------------
# Базы чатов
# --------------------------------------------------------------------------

def test_chat_database_visible_to_member(client, chat_database):
    names = [item["name"] for item in client.get("/api/databases").json()]

    assert "Чат путешественников" in names
    assert client.get(f"/api/databases/{chat_database['id']}").status_code == 200


def test_chat_database_hidden_from_outsider(client, chat_database):
    outsider = _other_user("postoronniy")
    try:
        assert [d["id"] for d in outsider.get("/api/databases").json()] == []
        assert outsider.get(f"/api/databases/{chat_database['id']}").status_code == 404
    finally:
        outsider.__exit__(None, None, None)


def test_member_added_later_gains_access(client, chat_database):
    """Участник появляется в базе, когда бот впервые видит его сообщение в чате."""
    newcomer = _other_user("novichok")
    try:
        assert newcomer.get(f"/api/databases/{chat_database['id']}").status_code == 404

        user = db.get_user_by_login("novichok")
        db.remember_chat_member("-100500", user["id"])

        assert newcomer.get(f"/api/databases/{chat_database['id']}").status_code == 200
    finally:
        newcomer.__exit__(None, None, None)


def test_member_who_left_loses_access(client, chat_database):
    """
    Проверяем именно участника, а не создателя: у создателя доступ остаётся по праву
    владения базой, и выход из чата его не отнимает.
    """
    leaver = _other_user("ushedshiy")
    try:
        user = db.get_user_by_login("ushedshiy")
        db.remember_chat_member("-100500", user["id"])
        assert leaver.get(f"/api/databases/{chat_database['id']}").status_code == 200

        db.forget_chat_member("-100500", user["id"])

        assert leaver.get(f"/api/databases/{chat_database['id']}").status_code == 404
        assert [d["id"] for d in leaver.get("/api/databases").json()] == []
    finally:
        leaver.__exit__(None, None, None)


def test_member_can_search_and_fill_chat_database(client, chat_database):
    files = [("files", ("a.jpg", _jpeg(), "image/jpeg"))]
    added = client.post(f"/api/databases/{chat_database['id']}/photos", files=files)

    assert added.status_code == 202
    assert added.json()["added"] == 1

    found = client.post(
        f"/api/databases/{chat_database['id']}/search/text",
        json={"query": "test", "translate": False},
    )
    assert len(found.json()["results"]) == 1


def test_chat_database_not_counted_in_personal_quota(client, chat_database, app_env):
    """Квота считает то, что человек создал сам: чат не должен съедать его лимит."""
    app_env.setenv("MAX_DB_PER_USER", "1")
    reset_settings()

    assert client.get("/api/quota").json()["databases_used"] == 0
    assert client.post("/api/databases", json={"name": "Личная"}).status_code == 201


# --------------------------------------------------------------------------
# Демо-база
# --------------------------------------------------------------------------

@pytest.fixture
def demo_database(client, registered, app_env, tmp_path, holder):
    """Имитация построенного CLI индекса COCO: старый формат meta плюс подписи."""
    index_dir = tmp_path / "demo-index"
    index_dir.mkdir()

    photo = tmp_path / "coco.jpg"
    photo.write_bytes(_jpeg((30, 60, 90)))

    vectors = np.ascontiguousarray(
        np.repeat([[1.0] + [0.0] * (holder.dim - 1)], 1, axis=0).astype("float32")
    )
    images = faiss.IndexFlatIP(holder.dim)
    images.add(vectors)
    faiss.write_index(images, str(index_dir / "images.index"))
    (index_dir / "images_meta.json").write_text(
        json.dumps([{"image_id": "139", "path": str(photo)}]), encoding="utf-8"
    )

    captions = faiss.IndexFlatIP(holder.dim)
    captions.add(vectors)
    faiss.write_index(captions, str(index_dir / "captions.index"))
    (index_dir / "captions_meta.json").write_text(
        json.dumps([{"image_id": "139", "caption": "A cat on a windowsill"}]), encoding="utf-8"
    )

    app_env.setenv("DEMO_INDEX_DIR", str(index_dir))
    reset_settings()

    from web.app import _register_demo_database

    _register_demo_database()
    return db.get_demo_database()


def test_demo_database_visible_to_everyone(client, demo_database):
    listed = client.get("/api/databases").json()

    demo = next(item for item in listed if item["id"] == demo_database["id"])
    assert demo["name"] == "Демо: MS COCO"
    assert listed[-1]["id"] == demo_database["id"]  # витрина идёт последней


def test_demo_database_searchable_with_captions(client, demo_database):
    """Единственная база, где работает поиск «фото → подпись»: у остальных подписей нет."""
    response = client.post(
        f"/api/databases/{demo_database['id']}/search/image",
        files={"file": ("q.jpg", _jpeg((30, 60, 90)), "image/jpeg")},
    )

    body = response.json()
    assert len(body["results"]) == 1
    assert body["captions"][0]["caption"] == "A cat on a windowsill"


def test_demo_photos_are_served(client, demo_database):
    """Снимки демо-базы лежат вне её папки — отдача не должна на этом спотыкаться."""
    photo_id = client.get(f"/api/databases/{demo_database['id']}/photos").json()["items"][0][
        "photo_id"
    ]

    response = client.get(f"/api/databases/{demo_database['id']}/photos/{photo_id}/file")

    assert response.status_code == 200
    assert len(response.content) > 0


def test_demo_database_cannot_be_changed(client, demo_database):
    demo_id = demo_database["id"]
    photo_id = client.get(f"/api/databases/{demo_id}/photos").json()["items"][0]["photo_id"]

    assert client.delete(f"/api/databases/{demo_id}/photos/{photo_id}").status_code == 403
    assert client.patch(f"/api/databases/{demo_id}", json={"name": "моя"}).status_code == 403
    assert client.delete(f"/api/databases/{demo_id}").status_code == 403
    assert client.post(
        f"/api/databases/{demo_id}/photos", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]
    ).status_code == 403


def test_demo_database_survives_delete_attempt(client, demo_database):
    """Отказ должен быть настоящим: файлы датасета на месте, база в списке."""
    from web.config import get_settings

    client.delete(f"/api/databases/{demo_database['id']}")

    assert (get_settings().demo_index_dir / "images.index").exists()
    assert db.get_demo_database() is not None


def test_demo_can_be_exported(client, demo_database):
    """Скачать витрину можно — это чтение, а не изменение."""
    response = client.get(f"/api/databases/{demo_database['id']}/export.zip")

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
