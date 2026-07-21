"""
Тесты хранения подписей (фаза C1).

Главное, что здесь проверяется, — то, чем индекс подписей отличается от индекса
картинок. У картинок i-я запись meta описывает i-й вектор, и это держится само
собой: снимок и его вектор появляются вместе. Подписи же приходят позже и
покрывают базу частично, поэтому у их индекса собственный список photo_id.

Позиционная адресация сломалась бы здесь не падением, а тихой подменой: удалили
снимок без подписи — и все последующие подписи съехали на одну строку, приклеившись
к чужим фотографиям. Поиск продолжил бы работать и молча врать, поэтому на этот
случай тестов больше всего.
"""

from __future__ import annotations

import json

import faiss
import numpy as np
import pytest

from core.store import (
    CAPTIONS_SBERT_INDEX,
    CAPTIONS_SBERT_META,
    IMAGES_META,
    META_VERSION,
    IndexStore,
    StoreError,
)

DIM = 8


def vector(seed: int, dim: int = DIM) -> np.ndarray:
    """Детерминированный нормированный вектор — заменяет SBERT в тестах."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def store_with_photos(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    result = store.add_photos([make_image() for _ in range(5)])
    assert result.added_count == 5
    return store, [p.photo_id for p in result.added]


# --------------------------------------------------------------------------
# Тексты подписей
# --------------------------------------------------------------------------

def test_caption_texts_survive_reopen(store_with_photos, tmp_path):
    store, ids = store_with_photos

    changed = store.set_caption_texts({ids[0]: "рыжий кот", ids[2]: "красный автобус"},
                                      model="blip-base")
    assert changed == 2

    reopened = IndexStore.open(tmp_path / "db")
    assert reopened.caption_of(ids[0]) == "рыжий кот"
    assert reopened.caption_of(ids[2]) == "красный автобус"
    assert reopened.caption_of(ids[1]) == ""
    assert reopened.get_photo(ids[0]).caption_model == "blip-base"


def test_coverage_counts_only_captioned(store_with_photos):
    store, ids = store_with_photos
    assert store.captions_coverage() == (0, 5)

    store.set_caption_texts({ids[0]: "кот", ids[1]: "пёс"})

    assert store.captions_coverage() == (2, 5)
    assert store.stats().captions_count == 2
    assert {p.photo_id for p in store.photos_without_caption()} == set(ids[2:])


def test_caption_for_unknown_photo_is_ignored(store_with_photos):
    """Снимок могли удалить, пока его подпись считалась в фоне, — это не ошибка."""
    store, ids = store_with_photos

    changed = store.set_caption_texts({ids[0]: "кот", "нет-такого-id": "призрак"})

    assert changed == 1
    assert store.captions_coverage() == (1, 5)


def test_meta_stays_v2_until_a_caption_appears(store_with_photos, tmp_path):
    """
    База без подписей должна остаться читаемой для кода, собранного до C1:
    data/ общая между локальным запуском и контейнерами, и образ со старым
    core на meta v3 базу просто не откроет.
    """
    store, ids = store_with_photos
    path = tmp_path / "db" / IMAGES_META

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["version"] == 2
    assert all("caption" not in item for item in meta["photos"])

    store.set_caption_texts({ids[0]: "кот"})

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["version"] == META_VERSION
    assert sum("caption" in item for item in meta["photos"]) == 1


def test_v2_meta_still_opens(store_with_photos, tmp_path):
    """У пользователя уже есть базы в старом формате — миграции быть не должно."""
    path = tmp_path / "db" / IMAGES_META
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["version"] = 2
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    reopened = IndexStore.open(tmp_path / "db")

    assert len(reopened) == 5
    assert reopened.captions_coverage() == (0, 5)


# --------------------------------------------------------------------------
# Векторы подписей
# --------------------------------------------------------------------------

def test_search_captions_finds_photo(store_with_photos):
    store, ids = store_with_photos
    store.set_caption_texts({ids[1]: "красный автобус"})
    store.set_caption_vectors({ids[1]: vector(1)}, model="mini")

    hits = store.search_captions(vector(1), top_k=3)

    assert [h.photo_id for h in hits] == [ids[1]]
    assert hits[0].caption == "красный автобус"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_vectors_survive_reopen(store_with_photos, tmp_path):
    store, ids = store_with_photos
    store.set_caption_texts({ids[0]: "кот"})
    store.set_caption_vectors({ids[0]: vector(7)}, model="mini")

    reopened = IndexStore.open(tmp_path / "db")

    assert reopened.caption_index_model() == "mini"
    assert [h.photo_id for h in reopened.search_captions(vector(7))] == [ids[0]]


def test_vectors_merge_across_calls(store_with_photos):
    """Фоновый генератор дописывает подписи порциями, а не одним заходом."""
    store, ids = store_with_photos
    store.set_caption_vectors({ids[0]: vector(1)}, model="mini")
    store.set_caption_vectors({ids[3]: vector(2)}, model="mini")

    assert {h.photo_id for h in store.search_captions(vector(1), top_k=5)} == {ids[0], ids[3]}


def test_vector_replaced_not_duplicated(store_with_photos):
    store, ids = store_with_photos
    store.set_caption_vectors({ids[0]: vector(1)}, model="mini")
    store.set_caption_vectors({ids[0]: vector(2)}, model="mini")

    hits = store.search_captions(vector(2), top_k=5)

    assert [h.photo_id for h in hits] == [ids[0]]  # одна строка, а не две
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_vectors_for_unknown_photos_ignored(store_with_photos):
    store, ids = store_with_photos
    added = store.set_caption_vectors({ids[0]: vector(1), "чужой": vector(2)}, model="mini")
    assert added == 1


def test_mixed_dimensions_rejected(store_with_photos):
    store, ids = store_with_photos
    with pytest.raises(StoreError):
        store.set_caption_vectors({ids[0]: vector(1, 8), ids[1]: vector(2, 16)})


def test_changing_model_discards_old_space(store_with_photos):
    """Векторы двух разных энкодеров в одном индексе давали бы бессмысленные оценки."""
    store, ids = store_with_photos
    store.set_caption_vectors({ids[0]: vector(1, 8)}, model="mini")
    store.set_caption_vectors({ids[1]: vector(2, 16)}, model="large")

    hits = store.search_captions(vector(2, 16), top_k=5)

    assert [h.photo_id for h in hits] == [ids[1]]
    assert store.caption_index_model() == "large"


def test_query_of_wrong_dimension_is_rejected(store_with_photos):
    store, ids = store_with_photos
    store.set_caption_vectors({ids[0]: vector(1, 8)}, model="mini")

    with pytest.raises(StoreError):
        store.search_captions(vector(1, 16))


# --------------------------------------------------------------------------
# Удаление при частичном покрытии — ради этого всё и затевалось
# --------------------------------------------------------------------------

def test_delete_photo_without_caption_keeps_others_correct(store_with_photos):
    """
    Ровно тот случай, который сломала бы позиционная адресация: убираем снимок,
    у которого подписи нет, и проверяем, что чужие подписи не съехали.
    """
    store, ids = store_with_photos
    store.set_caption_texts({ids[3]: "красный автобус", ids[4]: "рыжий кот"})
    store.set_caption_vectors({ids[3]: vector(3), ids[4]: vector(4)}, model="mini")

    assert store.delete_photos([ids[0]]) == 1

    bus = store.search_captions(vector(3), top_k=1)
    cat = store.search_captions(vector(4), top_k=1)
    assert [h.photo_id for h in bus] == [ids[3]]
    assert bus[0].caption == "красный автобус"
    assert [h.photo_id for h in cat] == [ids[4]]
    assert cat[0].caption == "рыжий кот"


def test_delete_photo_with_caption_drops_its_vector(store_with_photos):
    store, ids = store_with_photos
    store.set_caption_texts({ids[1]: "кот", ids[2]: "пёс"})
    store.set_caption_vectors({ids[1]: vector(1), ids[2]: vector(2)}, model="mini")

    store.delete_photos([ids[1]])

    assert [h.photo_id for h in store.search_captions(vector(1), top_k=5)] == [ids[2]]
    assert store.captions_coverage() == (1, 4)


def test_deletion_keeps_index_and_rows_in_step(store_with_photos, tmp_path):
    store, ids = store_with_photos
    store.set_caption_vectors({pid: vector(i) for i, pid in enumerate(ids)}, model="mini")

    store.delete_photos([ids[0], ids[2]])

    index = faiss.read_index(str(tmp_path / "db" / CAPTIONS_SBERT_INDEX))
    meta = json.loads((tmp_path / "db" / CAPTIONS_SBERT_META).read_text(encoding="utf-8"))
    assert index.ntotal == len(meta["rows"]) == 3
    assert set(meta["rows"]) == {ids[1], ids[3], ids[4]}


# --------------------------------------------------------------------------
# Порча вторичного индекса не должна ронять базу
# --------------------------------------------------------------------------

def test_broken_caption_index_disables_search_but_keeps_base(store_with_photos, tmp_path):
    store, ids = store_with_photos
    store.set_caption_texts({ids[0]: "кот"})
    store.set_caption_vectors({ids[0]: vector(1)}, model="mini")

    # число строк разошлось с числом векторов — так выглядит недописанный файл
    path = tmp_path / "db" / CAPTIONS_SBERT_META
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["rows"] = meta["rows"] * 2
    path.write_text(json.dumps(meta), encoding="utf-8")

    reopened = IndexStore.open(tmp_path / "db")

    assert len(reopened) == 5                       # база открылась
    assert reopened.search_text("кот", translate=False)[1]  # поиск по картинкам жив
    assert reopened.search_captions(vector(1)) == []        # поиск по подписям выключен
    assert reopened.caption_of(ids[0]) == "кот"             # текст подписи на месте


def test_missing_caption_index_is_not_an_error(store_with_photos):
    store, _ = store_with_photos
    assert store.search_captions(vector(1)) == []
    assert store.caption_index_model() == ""


# --------------------------------------------------------------------------
# Снятие подписи — текст и вектор должны уходить вместе
# --------------------------------------------------------------------------

def test_clear_caption_removes_text_and_vector(store_with_photos):
    """
    Правка подписи человеком: обнулить один текст, оставив вектор, значило бы, что
    поиск по подписям продолжает находить снимок по уже стёртым словам.
    """
    store, ids = store_with_photos
    store.set_caption_texts({ids[1]: "красный автобус"})
    store.set_caption_vectors({ids[1]: vector(1)}, model="mini")
    assert store.search_captions(vector(1))  # находится

    assert store.clear_caption(ids[1]) is True

    assert store.caption_of(ids[1]) == ""
    assert store.search_captions(vector(1)) == []  # больше не находится
    assert store.captions_coverage() == (0, 5)


def test_clear_caption_survives_reopen(store_with_photos, tmp_path):
    store, ids = store_with_photos
    store.set_caption_texts({ids[0]: "кот"})
    store.set_caption_vectors({ids[0]: vector(2)}, model="mini")
    store.clear_caption(ids[0])

    reopened = IndexStore.open(tmp_path / "db")
    assert reopened.caption_of(ids[0]) == ""
    assert reopened.search_captions(vector(2)) == []


def test_clear_caption_when_nothing_to_clear(store_with_photos):
    store, ids = store_with_photos
    assert store.clear_caption(ids[0]) is False       # подписи и не было
    assert store.clear_caption("нет-такого") is False  # и снимка нет


def test_clearing_one_caption_keeps_the_others(store_with_photos):
    store, ids = store_with_photos
    store.set_caption_texts({ids[0]: "кот", ids[3]: "пёс"})
    store.set_caption_vectors({ids[0]: vector(1), ids[3]: vector(2)}, model="mini")

    store.clear_caption(ids[0])

    assert [h.photo_id for h in store.search_captions(vector(2))] == [ids[3]]
    assert store.captions_coverage() == (1, 5)
