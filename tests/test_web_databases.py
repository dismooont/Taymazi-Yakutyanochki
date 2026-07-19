"""
Тесты жизненного цикла пользовательских баз.

Отдельное внимание — изоляции: чужая база должна быть неотличима от несуществующей,
иначе по кодам ответов перебираются id соседей.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from web.config import get_settings, reset_settings


def _second_user(login: str = "petr") -> TestClient:
    """Ещё один вошедший пользователь в том же приложении (та же папка данных и БД)."""
    from web.app import create_app

    client = TestClient(create_app())
    client.headers["X-Requested-With"] = "XMLHttpRequest"
    client.__enter__()
    response = client.post(
        "/api/auth/register", json={"login": login, "password": "korrektnyy-parol"}
    )
    assert response.status_code == 201, response.text
    return client


# --------------------------------------------------------------------------
# Создание
# --------------------------------------------------------------------------

def test_create_database(client, registered):
    response = client.post("/api/databases", json={"name": "Отпуск 2024"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Отпуск 2024"
    assert body["photos_count"] == 0
    assert body["photos_bytes"] == 0
    assert body["status"] == "ready"
    assert body["total_bytes"] == body["photos_bytes"] + body["index_bytes"]


def test_created_database_has_files_on_disk(client, registered):
    database = client.post("/api/databases", json={"name": "Пустая"}).json()

    root: Path = get_settings().database_dir(registered["id"], database["id"])
    assert (root / "images.index").exists()
    assert (root / "images_meta.json").exists()
    assert (root / "photos").is_dir()


def test_list_databases(client, registered):
    client.post("/api/databases", json={"name": "Первая"})
    client.post("/api/databases", json={"name": "Вторая"})

    names = [item["name"] for item in client.get("/api/databases").json()]

    assert sorted(names) == ["Вторая", "Первая"]


def test_database_limit_enforced(client, registered, app_env):
    app_env.setenv("MAX_DB_PER_USER", "2")
    reset_settings()

    assert client.post("/api/databases", json={"name": "1"}).status_code == 201
    assert client.post("/api/databases", json={"name": "2"}).status_code == 201
    response = client.post("/api/databases", json={"name": "3"})

    assert response.status_code == 409
    assert "лимит баз" in response.json()["detail"]


def test_quota_reports_usage(client, registered, app_env):
    app_env.setenv("MAX_DB_PER_USER", "3")
    reset_settings()
    client.post("/api/databases", json={"name": "Первая"})

    quota = client.get("/api/quota").json()

    assert quota["databases_used"] == 1
    assert quota["databases_limit"] == 3
    assert quota["bytes_limit"] > 0


# --------------------------------------------------------------------------
# Изменение и удаление
# --------------------------------------------------------------------------

def test_rename_database(client, registered):
    database = client.post("/api/databases", json={"name": "Старое"}).json()

    renamed = client.patch(f"/api/databases/{database['id']}", json={"name": "Новое"})

    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Новое"
    assert client.get(f"/api/databases/{database['id']}").json()["name"] == "Новое"


def test_delete_database_removes_files(client, registered):
    database = client.post("/api/databases", json={"name": "На удаление"}).json()
    root: Path = get_settings().database_dir(registered["id"], database["id"])
    assert root.exists()

    assert client.delete(f"/api/databases/{database['id']}").status_code == 204

    assert not root.exists()
    assert client.get(f"/api/databases/{database['id']}").status_code == 404
    assert client.get("/api/databases").json() == []


def test_stats_recomputed_from_disk(client, registered):
    database = client.post("/api/databases", json={"name": "Пустая"}).json()

    stats = client.get(f"/api/databases/{database['id']}/stats").json()

    assert stats["photos_count"] == 0
    assert stats["index_bytes"] > 0  # даже пустая база занимает место под индекс и meta
    assert stats["has_captions"] is False


# --------------------------------------------------------------------------
# Изоляция пользователей
# --------------------------------------------------------------------------

def test_databases_require_authentication(client):
    assert client.get("/api/databases").status_code == 401
    assert client.post("/api/databases", json={"name": "x"}).status_code == 401


def test_user_sees_only_own_databases(client, registered):
    client.post("/api/databases", json={"name": "Моя"})
    other = _second_user()
    try:
        other.post("/api/databases", json={"name": "Чужая"})

        assert [d["name"] for d in client.get("/api/databases").json()] == ["Моя"]
        assert [d["name"] for d in other.get("/api/databases").json()] == ["Чужая"]
    finally:
        other.__exit__(None, None, None)


def test_foreign_database_looks_nonexistent(client, registered):
    """
    Чужая база отдаёт 404, а не 403: по разнице кодов можно было бы узнать, какие id
    существуют, и заодно сколько баз у соседа.
    """
    victim = client.post("/api/databases", json={"name": "Секретная"}).json()
    attacker = _second_user("mallory")
    try:
        assert attacker.get(f"/api/databases/{victim['id']}").status_code == 404
        assert attacker.get(f"/api/databases/{victim['id']}/stats").status_code == 404
        assert attacker.patch(
            f"/api/databases/{victim['id']}", json={"name": "взломано"}
        ).status_code == 404
        assert attacker.delete(f"/api/databases/{victim['id']}").status_code == 404
    finally:
        attacker.__exit__(None, None, None)

    # база цела и не переименована
    assert client.get(f"/api/databases/{victim['id']}").json()["name"] == "Секретная"


def test_unknown_database_id_is_404(client, registered):
    assert client.get("/api/databases/nosuchid").status_code == 404
