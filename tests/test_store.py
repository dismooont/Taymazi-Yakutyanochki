"""
Тесты IndexStore — того, чего не умел прежний SearchEngine: пустая база, пакетное
добавление, удаление и учёт объёма.

Главный инвариант, ради которого написана половина этих тестов: i-я запись в
images_meta.json описывает i-й вектор в images.index. Если он нарушится, поиск начнёт
молча возвращать не те картинки — без падений и без ошибок в логах.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from PIL import Image

from core.embeddings import open_image
from core.store import (
    IMAGES_INDEX,
    IMAGES_META,
    META_VERSION,
    IndexStore,
    StoreError,
)


# --------------------------------------------------------------------------
# Пустая база
# --------------------------------------------------------------------------

def test_create_empty(tmp_path, holder):
    store = IndexStore.create_empty(tmp_path / "db")

    assert len(store) == 0
    assert (tmp_path / "db" / IMAGES_INDEX).exists()
    assert (tmp_path / "db" / IMAGES_META).exists()

    meta = json.loads((tmp_path / "db" / IMAGES_META).read_text(encoding="utf-8"))
    assert meta["version"] == META_VERSION
    assert meta["photos"] == []


def test_search_in_empty_base_returns_nothing(tmp_path, holder):
    """Прежний код на пустой базе падал: FAISS возвращает -1, а meta[-1] — последний элемент."""
    store = IndexStore.create_empty(tmp_path / "db")

    used_query, hits = store.search_text("рыжий кот", top_k=5, translate=False)
    assert hits == []
    assert used_query == "рыжий кот"

    stats = store.stats()
    assert stats.photos_count == 0
    assert stats.photos_bytes == 0
    assert stats.has_captions is False


def test_create_empty_twice_fails(tmp_path, holder):
    IndexStore.create_empty(tmp_path / "db")
    with pytest.raises(StoreError):
        IndexStore.create_empty(tmp_path / "db")


# --------------------------------------------------------------------------
# Добавление
# --------------------------------------------------------------------------

def test_add_photos(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    files = [make_image() for _ in range(3)]

    result = store.add_photos(files)

    assert result.added_count == 3
    assert result.skipped == []
    assert len(store) == 3

    for photo in result.added:
        assert store.photo_path(photo).exists()
        assert store.thumb_path(photo) is not None  # превью создано
        assert photo.bytes > 0

    stats = store.stats()
    assert stats.photos_count == 3
    assert stats.photos_bytes == sum(p.bytes for p in result.added)
    assert stats.index_bytes > 0


def test_add_reports_progress(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    files = [make_image() for _ in range(5)]

    seen: list[tuple[int, int]] = []
    store.add_photos(files, on_progress=lambda done, total: seen.append((done, total)))

    assert seen, "прогресс не приходил — веб не сможет показать индикатор"
    assert seen[-1] == (5, 5)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_add_skips_duplicates(tmp_path, holder, make_image):
    """photo_id считается от содержимого, поэтому тот же снимок под другим именем — дубль."""
    store = IndexStore.create_empty(tmp_path / "db")
    original = make_image(color=(10, 20, 30))
    copy = original.parent / "renamed.jpg"
    copy.write_bytes(original.read_bytes())

    first = store.add_photos([original])
    second = store.add_photos([copy])

    assert first.added_count == 1
    assert second.added_count == 0
    assert second.skipped == [("renamed.jpg", "уже есть в базе")]
    assert len(store) == 1


def test_add_skips_broken_files(tmp_path, holder, make_image):
    """Одна битая картинка в архиве на 1000 фото не должна отменять весь импорт."""
    store = IndexStore.create_empty(tmp_path / "db")
    good = make_image()

    not_an_image = tmp_path / "src" / "fake.jpg"
    not_an_image.write_bytes(b"NOT A JPEG AT ALL")
    wrong_ext = tmp_path / "src" / "notes.txt"
    wrong_ext.write_text("hello")

    result = store.add_photos([good, not_an_image, wrong_ext])

    assert result.added_count == 1
    assert len(store) == 1
    reasons = dict(result.skipped)
    assert reasons["fake.jpg"] == "не является изображением"
    assert reasons["notes.txt"] == "неподдерживаемый формат"


def test_reopen_after_add(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    files = [make_image() for _ in range(4)]
    store.add_photos(files)

    reopened = IndexStore.open(tmp_path / "db")

    assert len(reopened) == 4
    assert {p.photo_id for p in reopened.list_photos()} == {p.photo_id for p in store.list_photos()}


def test_open_detects_corrupted_base(tmp_path, holder, make_image):
    """Рассинхрон meta и индекса должен быть громкой ошибкой, а не тихо неверным поиском."""
    root = tmp_path / "db"
    store = IndexStore.create_empty(root)
    store.add_photos([make_image() for _ in range(3)])

    meta = json.loads((root / IMAGES_META).read_text(encoding="utf-8"))
    meta["photos"].pop()
    (root / IMAGES_META).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(StoreError, match="повреждена"):
        IndexStore.open(root)


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------

def test_search_image_finds_itself_first(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    files = [make_image() for _ in range(5)]
    added = store.add_photos(files).added

    target_file, target_photo = files[2], added[2]
    hits, captions = store.search_image(open_image(target_file), top_k=3)

    assert hits[0].photo_id == target_photo.photo_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert captions == []  # у пользовательской базы индекса подписей нет


def test_search_top_k_larger_than_base(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    store.add_photos([make_image() for _ in range(2)])

    _, hits = store.search_text("что угодно", top_k=50, translate=False)

    assert len(hits) == 2  # ровно столько, сколько есть, без -1 в выдаче


def test_search_hits_point_to_existing_files(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    store.add_photos([make_image() for _ in range(3)])

    _, hits = store.search_text("кот", top_k=3, translate=False)

    assert len(hits) == 3
    for hit in hits:
        assert Path(hit.path).exists()


# --------------------------------------------------------------------------
# Удаление — самое важное
# --------------------------------------------------------------------------

def test_delete_removes_photo_and_files(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    added = store.add_photos([make_image() for _ in range(3)]).added
    victim = added[1]
    victim_path = store.photo_path(victim)
    victim_thumb = store.thumb_path(victim)

    removed = store.delete_photos([victim.photo_id])

    assert removed == 1
    assert len(store) == 2
    assert not store.has(victim.photo_id)
    assert not victim_path.exists()
    assert not victim_thumb.exists()


def test_delete_keeps_index_and_meta_aligned(tmp_path, holder, make_image):
    """
    После удаления каждая оставшаяся картинка обязана находить саму себя с score ≈ 1.0.
    Если бы индекс пересобирался неправильно, поиск возвращал бы чужие photo_id —
    молча, без единой ошибки.
    """
    store = IndexStore.create_empty(tmp_path / "db")
    files = [make_image() for _ in range(6)]
    added = store.add_photos(files).added

    store.delete_photos([added[0].photo_id, added[3].photo_id, added[5].photo_id])

    survivors = [(files[i], added[i]) for i in (1, 2, 4)]
    assert len(store) == len(survivors)

    for file_path, photo in survivors:
        hits, _ = store.search_image(open_image(file_path), top_k=1)
        assert hits[0].photo_id == photo.photo_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_delete_survives_reopen(tmp_path, holder, make_image):
    root = tmp_path / "db"
    store = IndexStore.create_empty(root)
    added = store.add_photos([make_image() for _ in range(4)]).added
    store.delete_photos([added[0].photo_id])

    reopened = IndexStore.open(root)  # open() сам проверит соответствие meta и индекса

    assert len(reopened) == 3
    assert not reopened.has(added[0].photo_id)


def test_delete_unknown_id_is_noop(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    store.add_photos([make_image()])

    assert store.delete_photos(["нет-такого"]) == 0
    assert len(store) == 1


def test_delete_all_photos(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    added = store.add_photos([make_image() for _ in range(3)]).added

    removed = store.delete_photos([p.photo_id for p in added])

    assert removed == 3
    assert len(store) == 0
    _, hits = store.search_text("кот", top_k=5, translate=False)
    assert hits == []


def test_deleted_photo_can_be_added_again(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    image = make_image(color=(200, 100, 50))
    photo = store.add_photos([image]).added[0]
    store.delete_photos([photo.photo_id])

    result = store.add_photos([image])

    assert result.added_count == 1
    assert len(store) == 1


# --------------------------------------------------------------------------
# Экспорт
# --------------------------------------------------------------------------

def test_export_lists_all_files(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    store.add_photos([make_image() for _ in range(3)])

    entries = list(store.iter_export_files())

    assert len(entries) == 3
    for arcname, path in entries:
        assert arcname.startswith("photos/")
        assert path.exists()
    assert len(store.manifest()["photos"]) == 3


# --------------------------------------------------------------------------
# Совместимость со старым форматом (база COCO из index/)
# --------------------------------------------------------------------------

def _write_legacy_base(root: Path, image_paths: list[Path], dim: int) -> None:
    """Имитирует базу, построенную командой build до рефакторинга."""
    root.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(image_paths), dim)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(root / IMAGES_INDEX))
    (root / IMAGES_META).write_text(
        json.dumps([
            # обратные слеши: индекс мог быть построен на Windows
            {"image_id": str(i).zfill(12), "path": str(p).replace("/", "\\")}
            for i, p in enumerate(image_paths)
        ]),
        encoding="utf-8",
    )


def test_open_legacy_base(tmp_path, holder, make_image):
    files = [make_image() for _ in range(3)]
    root = tmp_path / "legacy"
    _write_legacy_base(root, files, holder.dim)

    store = IndexStore.open(root)

    assert len(store) == 3
    _, hits = store.search_text("кот", top_k=2, translate=False)
    assert len(hits) == 2
    assert Path(hits[0].path).exists()  # путь из legacy-meta разрешился в реальный файл


def test_legacy_base_is_read_only(tmp_path, holder, make_image):
    """
    Старую базу COCO веб может показывать, но не менять: у её записей нет ни photo_id
    по содержимому, ни файлов внутри папки базы.
    """
    files = [make_image() for _ in range(2)]
    root = tmp_path / "legacy"
    _write_legacy_base(root, files, holder.dim)
    store = IndexStore.open(root)

    with pytest.raises(StoreError, match="только для чтения"):
        store.add_photos([make_image()])
    with pytest.raises(StoreError, match="только для чтения"):
        store.delete_photos([store.list_photos()[0].photo_id])
