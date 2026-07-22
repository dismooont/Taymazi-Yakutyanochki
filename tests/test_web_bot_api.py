"""
Тесты ручек, которыми пользуется Telegram-бот.

Бот — доверенный внутренний клиент: он сообщает, от чьего имени действует, и API
ему верит. Поэтому проверок здесь две группы: что без служебного токена внутрь не
попасть вовсе и что действия бота подчиняются тем же правилам, что и действия
с сайта — квотам, запрету на изменение демо-базы, учёту участников чата.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from web import db
from web.config import reset_settings

TOKEN = "sluzhebnyy-token-dlya-testov-0123456789"
CHAT = "-100777"


def _jpeg(color=(120, 80, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def bot_client(raw_client, app_env):
    """Клиент со служебным токеном — то, чем пользуется процесс бота."""
    app_env.setenv("SERVICE_TOKEN", TOKEN)
    reset_settings()
    raw_client.headers["X-Service-Token"] = TOKEN
    return raw_client


@pytest.fixture
def demo_ready(bot_client, app_env, tmp_path, holder):
    """Подключает демо-базу так же, как это делает приложение при старте."""
    import json

    import faiss
    import numpy as np

    index_dir = tmp_path / "demo-for-bot"
    index_dir.mkdir()
    photo = tmp_path / "coco.jpg"
    photo.write_bytes(_jpeg((30, 60, 90)))

    vectors = np.ascontiguousarray(
        np.array([[1.0] + [0.0] * (holder.dim - 1)], dtype="float32")
    )
    index = faiss.IndexFlatIP(holder.dim)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "images.index"))
    (index_dir / "images_meta.json").write_text(
        json.dumps([{"image_id": "139", "path": str(photo)}]), encoding="utf-8"
    )

    app_env.setenv("DEMO_INDEX_DIR", str(index_dir))
    reset_settings()

    from web.app import _register_demo_database

    _register_demo_database()
    return db.get_demo_database()


@pytest.fixture
def started_chat(bot_client):
    response = bot_client.post(
        f"/api/bot/chats/{CHAT}/start",
        data={"telegram_user_id": "555", "display_name": "Иван", "title": "Поход"},
    )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Служебный токен
# --------------------------------------------------------------------------

def test_without_token_no_access(raw_client, app_env):
    app_env.setenv("SERVICE_TOKEN", TOKEN)
    reset_settings()

    response = raw_client.post(
        f"/api/bot/chats/{CHAT}/start", data={"telegram_user_id": "555"}
    )

    assert response.status_code == 401


def test_wrong_token_rejected(raw_client, app_env):
    app_env.setenv("SERVICE_TOKEN", TOKEN)
    reset_settings()

    response = raw_client.post(
        f"/api/bot/chats/{CHAT}/start",
        data={"telegram_user_id": "555"},
        headers={"X-Service-Token": "ne-tot-token"},
    )

    assert response.status_code == 401


def test_endpoints_absent_until_token_configured(raw_client):
    """Пока SERVICE_TOKEN не задан, ручек бота как будто нет — так безопаснее умолчания."""
    response = raw_client.post(
        f"/api/bot/chats/{CHAT}/start",
        data={"telegram_user_id": "555"},
        headers={"X-Service-Token": "chto-ugodno"},
    )

    assert response.status_code == 404


def test_bot_requests_need_no_csrf_header(bot_client, started_chat):
    """
    Бот не браузер и авторизуется токеном, а не cookie: требовать X-Requested-With
    у него означало бы заставлять слать бессмысленную строку.
    """
    assert "X-Requested-With" not in bot_client.headers
    assert started_chat["photos_count"] == 0


# --------------------------------------------------------------------------
# Чат и его база
# --------------------------------------------------------------------------

def test_start_creates_database_once(bot_client, started_chat):
    assert started_chat["created"] is True
    assert started_chat["name"] == "Поход"

    again = bot_client.post(
        f"/api/bot/chats/{CHAT}/start",
        data={"telegram_user_id": "555", "display_name": "Иван", "title": "Поход"},
    ).json()

    assert again["created"] is False
    assert again["database_id"] == started_chat["database_id"]


def test_chat_without_start_is_unknown(bot_client):
    assert bot_client.get("/api/bot/chats/-100999").status_code == 404


def test_start_creates_account_linked_to_telegram(bot_client, started_chat):
    """
    Аккаунт из чата — тот же, в который человек попадёт, войдя на сайт через
    Telegram: иначе снимки из бота не нашлись бы в веб-интерфейсе.
    """
    user = db.get_user_by_identity("telegram", "555")

    assert user is not None
    assert user["display_name"] == "Иван"
    assert db.get_chat_member(CHAT, user["id"]) is not None


def test_member_recorded_on_message(bot_client, started_chat):
    bot_client.post(
        f"/api/bot/chats/{CHAT}/members",
        data={"telegram_user_id": "666", "display_name": "Пётр"},
    )

    petr = db.get_user_by_identity("telegram", "666")
    assert db.get_chat_member(CHAT, petr["id"]) is not None


# --------------------------------------------------------------------------
# Фотографии и поиск
# --------------------------------------------------------------------------

def test_add_photo_and_search(bot_client, started_chat):
    added = bot_client.post(
        f"/api/bot/chats/{CHAT}/photos",
        files={"file": ("a.jpg", _jpeg(), "image/jpeg")},
        data={"telegram_user_id": "555", "display_name": "Иван"},
    )

    assert added.status_code == 200
    assert added.json()["added"] == 1
    assert added.json()["photos_count"] == 1

    found = bot_client.post(
        f"/api/bot/chats/{CHAT}/search", json={"query": "cat", "top_k": 3, "translate": False}
    ).json()

    assert len(found["results"]) == 1
    photo_id = found["results"][0]["photo_id"]
    file_response = bot_client.get(f"/api/bot/chats/{CHAT}/photos/{photo_id}/file")
    assert file_response.status_code == 200
    assert len(file_response.content) > 0


def test_duplicate_photo_reported_as_skipped(bot_client, started_chat):
    photo = _jpeg((7, 7, 7))
    payload = {"file": ("a.jpg", photo, "image/jpeg")}
    bot_client.post(f"/api/bot/chats/{CHAT}/photos", files=payload)

    again = bot_client.post(
        f"/api/bot/chats/{CHAT}/photos", files={"file": ("b.jpg", photo, "image/jpeg")}
    ).json()

    assert again["added"] == 0
    assert again["skipped"][0][1] == "уже есть в базе"


def test_photo_quota_applies_to_bot(bot_client, started_chat, app_env):
    """Ключевая причина ходить через API: правила одни и те же у бота и у сайта."""
    app_env.setenv("MAX_PHOTOS_PER_DB", "1")
    reset_settings()

    bot_client.post(f"/api/bot/chats/{CHAT}/photos", files={"file": ("a.jpg", _jpeg((1, 2, 3)), "image/jpeg")})
    second = bot_client.post(
        f"/api/bot/chats/{CHAT}/photos", files={"file": ("b.jpg", _jpeg((9, 9, 9)), "image/jpeg")}
    )

    assert second.status_code == 409
    assert "не более 1 фото" in second.json()["detail"]


def test_search_without_database(bot_client):
    response = bot_client.post(
        "/api/bot/chats/-100999/search", json={"query": "cat", "translate": False}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Связь с веб-интерфейсом
# --------------------------------------------------------------------------

def test_photos_from_bot_visible_on_site(bot_client, started_chat, client, app_env):
    """
    Снимок, присланный в чат, должен находиться в веб-интерфейсе тем же человеком:
    ради этого бот и сайт работают с одной базой через один API.
    """
    bot_client.post(
        f"/api/bot/chats/{CHAT}/photos",
        files={"file": ("a.jpg", _jpeg((40, 90, 140)), "image/jpeg")},
        data={"telegram_user_id": "555", "display_name": "Иван"},
    )

    # тот же человек входит на сайте через Telegram.
    # Настройки правим только через app_env: os.environ напрямую пережил бы тест
    # и сломал следующий (именно так и случилось при первом прогоне).
    from tests.test_web_telegram import BOT_TOKEN, widget_payload

    app_env.setenv("TELEGRAM_AUTH_ENABLED", "1")
    app_env.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    app_env.setenv("TELEGRAM_BOT_USERNAME", "test_search_bot")
    reset_settings()

    client.headers["X-Requested-With"] = "XMLHttpRequest"
    entered = client.post("/api/auth/telegram", json=widget_payload("555"))
    assert entered.status_code == 200

    listed = client.get("/api/databases").json()
    chat_base = next(item for item in listed if item["id"] == started_chat["database_id"])
    assert chat_base["photos_count"] == 1

    found = client.post(
        f"/api/databases/{chat_base['id']}/search/text",
        json={"query": "cat", "translate": False},
    ).json()
    assert len(found["results"]) == 1


# --------------------------------------------------------------------------
# Управление снимками из чата: паритет с сайтом (удаление, подпись, экспорт)
# --------------------------------------------------------------------------

def _add_photo(bot_client, color) -> str:
    """Добавляет снимок в базу чата и возвращает его photo_id."""
    r = bot_client.post(
        f"/api/bot/chats/{CHAT}/photos",
        files={"file": ("p.jpg", _jpeg(color), "image/jpeg")},
        data={"telegram_user_id": "555", "display_name": "Иван"},
    )
    assert r.status_code == 200, r.text
    return r.json()["photo_id"]


def test_bot_can_delete_photo(bot_client, started_chat):
    pid = _add_photo(bot_client, (10, 20, 30))

    resp = bot_client.delete(f"/api/bot/chats/{CHAT}/photos/{pid}")
    assert resp.status_code == 204

    assert bot_client.get(f"/api/bot/chats/{CHAT}").json()["photos_count"] == 0
    # повторное удаление того же — уже 404
    assert bot_client.delete(f"/api/bot/chats/{CHAT}/photos/{pid}").status_code == 404


def test_bot_can_set_and_clear_caption(bot_client, started_chat):
    pid = _add_photo(bot_client, (40, 50, 60))

    resp = bot_client.put(
        f"/api/bot/chats/{CHAT}/photos/{pid}/caption", json={"caption": "рыжий кот"}
    )
    assert resp.status_code == 200
    assert resp.json()["caption"] == "рыжий кот"

    # подпись видна и на сайте — одна база
    assert bot_client.get(f"/api/bot/chats/{CHAT}").json()["captions_count"] == 1

    cleared = bot_client.put(
        f"/api/bot/chats/{CHAT}/photos/{pid}/caption", json={"caption": ""}
    )
    assert cleared.json()["caption"] == ""
    assert bot_client.get(f"/api/bot/chats/{CHAT}").json()["captions_count"] == 0


def test_bot_caption_unknown_photo_404(bot_client, started_chat):
    assert bot_client.put(
        f"/api/bot/chats/{CHAT}/photos/нет-такого/caption", json={"caption": "x"}
    ).status_code == 404


def test_bot_export_returns_zip(bot_client, started_chat):
    import io as _io
    import zipfile

    _add_photo(bot_client, (11, 22, 33))
    _add_photo(bot_client, (99, 88, 77))

    resp = bot_client.get(f"/api/bot/chats/{CHAT}/export.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
        names = z.namelist()
    # два снимка плюс манифест базы
    assert sum(1 for n in names if n.startswith("photos/")) == 2
    assert any("manifest" in n for n in names)


def test_bot_import_zip(bot_client, started_chat):
    """Импорт группы фото из zip: ставится задача, по ней снимки доезжают в базу."""
    import io as _io
    import time
    import zipfile

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(4):
            z.writestr(f"photos/i{i}.jpg", _jpeg((i * 40 % 256, i * 30 % 256, i * 20 % 256)))

    started = bot_client.post(
        f"/api/bot/chats/{CHAT}/import",
        files={"file": ("a.zip", buf.getvalue(), "application/zip")},
    )
    assert started.status_code == 200, started.text
    assert started.json()["count"] == 4
    job_id = started.json()["job_id"]

    for _ in range(100):
        job = bot_client.get(f"/api/bot/chats/{CHAT}/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job

    assert bot_client.get(f"/api/bot/chats/{CHAT}").json()["photos_count"] == 4


def test_bot_import_rejects_non_zip(bot_client, started_chat):
    resp = bot_client.post(
        f"/api/bot/chats/{CHAT}/import",
        files={"file": ("notzip.jpg", _jpeg(), "image/jpeg")},
    )
    assert resp.status_code == 422  # inspect не признаёт это архивом


def test_bot_job_status_unknown_is_404(bot_client, started_chat):
    assert bot_client.get(f"/api/bot/chats/{CHAT}/jobs/нет-такой").status_code == 404


# --------------------------------------------------------------------------
# Демо-база через бота
# --------------------------------------------------------------------------

def test_demo_absent_until_registered(bot_client):
    """Пока демо-база не подключена, бот получает честный 404, а не пустую выдачу."""
    assert bot_client.get("/api/bot/demo").status_code == 404
    assert bot_client.post(
        "/api/bot/demo/search", json={"query": "cat", "translate": False}
    ).status_code == 404


def test_demo_search_and_photo(bot_client, demo_ready):
    info = bot_client.get("/api/bot/demo").json()
    assert info["photos_count"] == 1

    found = bot_client.post(
        "/api/bot/demo/search", json={"query": "cat", "top_k": 3, "translate": False}
    ).json()

    assert len(found["results"]) == 1
    photo_id = found["results"][0]["photo_id"]
    assert found["results"][0]["file_url"] == f"/api/bot/demo/photos/{photo_id}/file"

    photo = bot_client.get(f"/api/bot/demo/photos/{photo_id}/file")
    assert photo.status_code == 200
    assert len(photo.content) > 0


def test_demo_needs_service_token(raw_client, demo_ready):
    """Демо-база открыта на чтение всем вошедшим на сайт, но не всему интернету."""
    # фикстура demo_ready проставила заголовок на этот же клиент — снимаем его,
    # чтобы проверить именно отсутствие токена
    raw_client.headers.pop("X-Service-Token", None)

    assert raw_client.get("/api/bot/demo").status_code == 401


def test_demo_available_from_any_chat(bot_client, started_chat, demo_ready):
    """Демо не привязана к чату: она общая, /start для неё не нужен."""
    from_known = bot_client.post(
        "/api/bot/demo/search", json={"query": "cat", "translate": False}
    )
    assert from_known.status_code == 200
    # чата -100999 не существует, но демо всё равно доступна
    assert bot_client.get("/api/bot/chats/-100999").status_code == 404
    assert bot_client.get("/api/bot/demo").status_code == 200


def test_add_photo_returns_id_for_similar_search(bot_client, started_chat):
    """Бот получает id добавленного снимка, чтобы сразу спросить похожие на него."""
    added = bot_client.post(
        f"/api/bot/chats/{CHAT}/photos", files={"file": ("a.jpg", _jpeg((10, 20, 30)), "image/jpeg")}
    ).json()

    assert added["photo_id"]

    bot_client.post(
        f"/api/bot/chats/{CHAT}/photos", files={"file": ("b.jpg", _jpeg((200, 30, 40)), "image/jpeg")}
    )

    similar = bot_client.get(
        f"/api/bot/chats/{CHAT}/photos/{added['photo_id']}/similar", params={"top_k": 5}
    ).json()

    assert len(similar["results"]) == 1  # второй снимок, но не сам запрошенный
    assert similar["results"][0]["photo_id"] != added["photo_id"]


def test_similar_for_unknown_photo_is_empty(bot_client, started_chat):
    response = bot_client.get(f"/api/bot/chats/{CHAT}/photos/deadbeef/similar")

    assert response.status_code == 200
    assert response.json()["results"] == []
