"""
Тесты загрузки фотографий, импорта архива и очереди фоновых задач.
"""

from __future__ import annotations

import io
import time
import zipfile

import pytest
from PIL import Image


def _jpeg(color=(100, 150, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


def _files(count: int, prefix: str = "photo"):
    """multipart-поле files с N разными картинками."""
    return [
        ("files", (f"{prefix}{i}.jpg", _jpeg((i * 9 % 256, i * 17 % 256, i * 31 % 256)), "image/jpeg"))
        for i in range(count)
    ]


def _zip_of(count: int, folder: str = "photos") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(count):
            archive.writestr(
                f"{folder}/img{i}.jpg", _jpeg((i * 7 % 256, i * 13 % 256, i * 23 % 256))
            )
    return buffer.getvalue()


def _wait_job(client, job_id: str, timeout: float = 20.0) -> dict:
    """Ждёт завершения задачи так же, как это делает фронтенд, — опросом."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"Задача {job_id} не завершилась за {timeout} с")


@pytest.fixture
def database(client, registered):
    return client.post("/api/databases", json={"name": "Тестовая"}).json()


# --------------------------------------------------------------------------
# Синхронная загрузка (мало файлов)
# --------------------------------------------------------------------------

def test_upload_few_photos_is_synchronous(client, database):
    response = client.post(f"/api/databases/{database['id']}/photos", files=_files(2))

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] is None  # ради двух снимков очередь не нужна
    assert body["added"] == 2

    stats = client.get(f"/api/databases/{database['id']}/stats").json()
    assert stats["photos_count"] == 2
    assert stats["photos_bytes"] > 0


def test_uploaded_photos_are_listed(client, database):
    client.post(f"/api/databases/{database['id']}/photos", files=_files(3))

    page = client.get(f"/api/databases/{database['id']}/photos").json()

    assert page["total"] == 3
    assert len(page["items"]) == 3
    assert all(item["bytes"] > 0 for item in page["items"])


def test_duplicate_upload_is_skipped(client, database):
    same = [("files", ("a.jpg", _jpeg((5, 5, 5)), "image/jpeg"))]
    client.post(f"/api/databases/{database['id']}/photos", files=same)

    response = client.post(f"/api/databases/{database['id']}/photos", files=same)

    assert response.json()["added"] == 0
    assert response.json()["skipped"][0][1] == "уже есть в базе"
    assert client.get(f"/api/databases/{database['id']}/stats").json()["photos_count"] == 1


def test_broken_file_does_not_break_upload(client, database):
    files = _files(1) + [("files", ("broken.jpg", b"not an image", "image/jpeg"))]

    response = client.post(f"/api/databases/{database['id']}/photos", files=files)

    assert response.json()["added"] == 1
    assert response.json()["skipped"] == [["broken.jpg", "не является изображением"]]


# --------------------------------------------------------------------------
# Фоновая загрузка (много файлов)
# --------------------------------------------------------------------------

def test_many_photos_go_to_background_job(client, database):
    response = client.post(f"/api/databases/{database['id']}/photos", files=_files(6))

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert job_id  # ответ пришёл сразу, не дожидаясь индексации

    job = _wait_job(client, job_id)
    assert job["status"] == "done"
    assert job["progress_done"] == job["progress_total"] == 6
    assert client.get(f"/api/databases/{database['id']}/stats").json()["photos_count"] == 6


def test_job_visible_in_history(client, database):
    job_id = client.post(f"/api/databases/{database['id']}/photos", files=_files(5)).json()["job_id"]
    _wait_job(client, job_id)

    jobs = client.get("/api/jobs", params={"database_id": database["id"]}).json()

    assert [job["id"] for job in jobs] == [job_id]
    assert jobs[0]["kind"] == "add_photos"


def test_concurrent_operation_on_same_database_rejected(client, database):
    """Два прогресс-бара на одной базе только путают — второй запрос отклоняется."""
    client.post(f"/api/databases/{database['id']}/photos", files=_files(8))

    second = client.post(f"/api/databases/{database['id']}/photos", files=_files(4, "other"))

    assert second.status_code == 409
    assert "уже обрабатывается" in second.json()["detail"]


# --------------------------------------------------------------------------
# Импорт архива
# --------------------------------------------------------------------------

def test_import_zip(client, database):
    response = client.post(
        f"/api/databases/{database['id']}/import",
        files={"file": ("photos.zip", _zip_of(5), "application/zip")},
    )

    assert response.status_code == 202
    job = _wait_job(client, response.json()["job_id"])

    assert job["status"] == "done"
    assert "Добавлено фото: 5" in job["message"]
    assert client.get(f"/api/databases/{database['id']}/stats").json()["photos_count"] == 5


def test_import_rejects_non_zip(client, database):
    response = client.post(
        f"/api/databases/{database['id']}/import",
        files={"file": ("fake.zip", b"definitely not a zip", "application/zip")},
    )

    assert response.status_code == 422
    assert "не zip-архив" in response.json()["detail"]


def test_import_rejects_archive_without_images(client, database):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"hello")

    response = client.post(
        f"/api/databases/{database['id']}/import",
        files={"file": ("docs.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 422
    assert "нет изображений" in response.json()["detail"]


def test_import_leaves_no_temporary_files(client, database, registered):
    from web.config import get_settings

    job_id = client.post(
        f"/api/databases/{database['id']}/import",
        files={"file": ("photos.zip", _zip_of(4), "application/zip")},
    ).json()["job_id"]
    _wait_job(client, job_id)

    tmp = get_settings().database_dir(registered["id"], database["id"]) / "tmp"
    assert not tmp.exists() or not any(tmp.iterdir())


# --------------------------------------------------------------------------
# Отмена
# --------------------------------------------------------------------------

def test_cancel_finished_job_rejected(client, database):
    job_id = client.post(f"/api/databases/{database['id']}/photos", files=_files(5)).json()["job_id"]
    _wait_job(client, job_id)

    response = client.post(f"/api/jobs/{job_id}/cancel")

    assert response.status_code == 409
    assert "уже завершена" in response.json()["detail"]


def test_cancelled_job_stops_and_keeps_base_consistent(client, database):
    """
    Отмена кооперативная: часть фото может успеть проиндексироваться. Требование —
    не «ноль добавленных», а согласованная база: meta и индекс должны сойтись,
    что и проверяет повторное открытие базы через /stats.
    """
    job_id = client.post(f"/api/databases/{database['id']}/photos", files=_files(10)).json()["job_id"]
    client.post(f"/api/jobs/{job_id}/cancel")

    job = _wait_job(client, job_id)
    assert job["status"] in ("done", "error")

    stats = client.get(f"/api/databases/{database['id']}/stats").json()
    page = client.get(f"/api/databases/{database['id']}/photos", params={"limit": 200}).json()
    assert stats["photos_count"] == page["total"]


# --------------------------------------------------------------------------
# Удаление
# --------------------------------------------------------------------------

def test_delete_photo(client, database):
    client.post(f"/api/databases/{database['id']}/photos", files=_files(3))
    photos = client.get(f"/api/databases/{database['id']}/photos").json()["items"]
    victim = photos[0]["photo_id"]

    assert client.delete(f"/api/databases/{database['id']}/photos/{victim}").status_code == 204

    remaining = client.get(f"/api/databases/{database['id']}/photos").json()
    assert remaining["total"] == 2
    assert victim not in [p["photo_id"] for p in remaining["items"]]
    assert client.get(f"/api/databases/{database['id']}/stats").json()["photos_count"] == 2


def test_delete_unknown_photo_is_404(client, database):
    assert client.delete(f"/api/databases/{database['id']}/photos/nosuch").status_code == 404


def test_bulk_delete(client, database):
    client.post(f"/api/databases/{database['id']}/photos", files=_files(3))
    ids = [p["photo_id"] for p in client.get(f"/api/databases/{database['id']}/photos").json()["items"]]

    response = client.post(
        f"/api/databases/{database['id']}/photos/delete", json={"photo_ids": ids[:2]}
    )

    assert response.json() == {"deleted": 2}
    assert client.get(f"/api/databases/{database['id']}/stats").json()["photos_count"] == 1


# --------------------------------------------------------------------------
# Квоты и изоляция
# --------------------------------------------------------------------------

def test_photo_limit_enforced(client, database, app_env):
    from web.config import reset_settings

    app_env.setenv("MAX_PHOTOS_PER_DB", "2")
    reset_settings()

    response = client.post(f"/api/databases/{database['id']}/photos", files=_files(3))

    assert response.status_code == 409
    assert "не более 2 фото" in response.json()["detail"]


def test_foreign_database_cannot_be_filled(client, database):
    from tests.test_web_databases import _second_user

    attacker = _second_user("mallory2")
    try:
        response = attacker.post(f"/api/databases/{database['id']}/photos", files=_files(1))
        assert response.status_code == 404
    finally:
        attacker.__exit__(None, None, None)


def test_foreign_job_is_invisible(client, database):
    from tests.test_web_databases import _second_user

    job_id = client.post(f"/api/databases/{database['id']}/photos", files=_files(5)).json()["job_id"]
    attacker = _second_user("mallory3")
    try:
        assert attacker.get(f"/api/jobs/{job_id}").status_code == 404
        assert attacker.post(f"/api/jobs/{job_id}/cancel").status_code == 404
    finally:
        attacker.__exit__(None, None, None)
    _wait_job(client, job_id)
