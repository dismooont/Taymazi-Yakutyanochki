"""
Тесты выгрузки базы в zip.

Главная проверка — не «ответ 200», а что скачанный архив действительно открывается
и что его можно залить обратно, получив ту же базу. Потоковая сборка zip легко даёт
формально успешный ответ с испорченным центральным каталогом.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from PIL import Image


def _jpeg(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def filled(client, registered):
    database = client.post("/api/databases", json={"name": "Отпуск 2024"}).json()
    files = [
        ("files", (f"p{i}.jpg", _jpeg((i * 40 % 256, i * 70 % 256, i * 90 % 256)), "image/jpeg"))
        for i in range(3)
    ]
    client.post(f"/api/databases/{database['id']}/photos", files=files)
    return database


def _download(client, database_id: str) -> zipfile.ZipFile:
    response = client.get(f"/api/databases/{database_id}/export.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    return zipfile.ZipFile(io.BytesIO(response.content))


# --------------------------------------------------------------------------
# Содержимое архива
# --------------------------------------------------------------------------

def test_export_produces_valid_zip(client, filled):
    archive = _download(client, filled["id"])

    assert archive.testzip() is None  # контрольные суммы всех записей сошлись
    photos = [n for n in archive.namelist() if n.startswith("photos/")]
    assert len(photos) == 3


def test_export_contains_manifest(client, filled):
    archive = _download(client, filled["id"])

    manifest = json.loads(archive.read("manifest.json"))

    assert len(manifest["photos"]) == 3
    assert manifest["exported_at"]
    assert all(item["photo_id"] and item["bytes"] > 0 for item in manifest["photos"])


def test_exported_files_are_intact(client, filled):
    """Файлы в архиве должны побайтово совпадать с тем, что отдаёт API."""
    archive = _download(client, filled["id"])
    items = client.get(f"/api/databases/{filled['id']}/photos").json()["items"]

    for item in items:
        original = client.get(
            f"/api/databases/{filled['id']}/photos/{item['photo_id']}/file"
        ).content
        stored = next(
            archive.read(n) for n in archive.namelist() if item["photo_id"] in n
        )
        assert stored == original


def test_empty_database_exports_to_valid_zip(client, registered):
    database = client.post("/api/databases", json={"name": "Пустая"}).json()

    archive = _download(client, database["id"])

    assert archive.testzip() is None
    assert archive.namelist() == ["manifest.json"]
    assert json.loads(archive.read("manifest.json"))["photos"] == []


# --------------------------------------------------------------------------
# Круговой сценарий: выгрузил и залил обратно
# --------------------------------------------------------------------------

def test_export_can_be_imported_back(client, filled):
    """
    Выгруженный архив обязан приниматься собственным же импортом. photo_id — хеш
    содержимого, поэтому в новой базе идентификаторы должны получиться те же самые.
    """
    exported = client.get(f"/api/databases/{filled['id']}/export.zip").content
    source_ids = {
        item["photo_id"]
        for item in client.get(f"/api/databases/{filled['id']}/photos").json()["items"]
    }

    restored = client.post("/api/databases", json={"name": "Восстановленная"}).json()
    response = client.post(
        f"/api/databases/{restored['id']}/import",
        files={"file": ("backup.zip", exported, "application/zip")},
    )
    assert response.status_code == 202

    from tests.test_web_photos import _wait_job

    job = _wait_job(client, response.json()["job_id"])
    assert job["status"] == "done"

    restored_ids = {
        item["photo_id"]
        for item in client.get(f"/api/databases/{restored['id']}/photos").json()["items"]
    }
    assert restored_ids == source_ids


def test_restored_database_is_searchable(client, filled):
    """Восстановленная база должна не просто содержать файлы, а работать в поиске."""
    exported = client.get(f"/api/databases/{filled['id']}/export.zip").content
    restored = client.post("/api/databases", json={"name": "Восстановленная"}).json()
    job_id = client.post(
        f"/api/databases/{restored['id']}/import",
        files={"file": ("backup.zip", exported, "application/zip")},
    ).json()["job_id"]

    from tests.test_web_photos import _wait_job

    _wait_job(client, job_id)

    hits = client.post(
        f"/api/databases/{restored['id']}/search/image",
        files={"file": ("q.jpg", _jpeg((0, 70, 180)), "image/jpeg")},
    ).json()["results"]
    assert len(hits) == 3


# --------------------------------------------------------------------------
# Имя файла и доступ
# --------------------------------------------------------------------------

def test_cyrillic_database_name_in_filename(client, filled):
    """
    В заголовке HTTP разрешён только ASCII, поэтому кириллическое имя уходит
    в filename* по RFC 5987, а в filename остаётся безопасная замена.
    """
    response = client.get(f"/api/databases/{filled['id']}/export.zip")

    disposition = response.headers["content-disposition"]
    assert "filename*=UTF-8''" in disposition
    assert disposition.isascii()
    assert "2024" in disposition


def test_export_requires_ownership(client, filled):
    from tests.test_web_databases import _second_user

    attacker = _second_user("mallory5")
    try:
        assert attacker.get(f"/api/databases/{filled['id']}/export.zip").status_code == 404
    finally:
        attacker.__exit__(None, None, None)


def test_export_requires_authentication(raw_client, filled):
    raw_client.cookies.clear()
    assert raw_client.get(f"/api/databases/{filled['id']}/export.zip").status_code == 401
