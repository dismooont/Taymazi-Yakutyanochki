"""
Тесты слияния двух путей поиска (фаза C2).

Проверяется не «стало лучше» — это измерено в C0 на 2500 отложенных запросах и
записано в docs/CAPTION_SEARCH.md. Здесь проверяется, что рантайм делает ровно
то же, что делал замер, и не разваливается на краях.

Самое важное — поведение при неполном покрытии. База размечается постепенно, и
в середине разметки часть снимков подписи ещё не имеет. Тут возможны две порчи
выдачи, и обе тихие: снимки без подписи могут провалиться скопом вниз, а при
малом покрытии лучшая из трёх подписей получает высокую оценку просто потому,
что она лучшая из трёх. Тесты ниже стерегут обе.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.store import (
    CAPTION_FUSION_ALPHA,
    CAPTION_FUSION_MIN_COVERAGE,
    CAPTION_FUSION_MIN_PHOTOS,
    IndexStore,
    StoreError,
)

DIM = 8
# Вдвое больше порога CAPTION_FUSION_MIN_PHOTOS: половина базы тоже должна его
# перекрывать, иначе нечем проверить включение слияния на половинном покрытии.
SIZE = 24


def vector(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


def encoder_for(vec: np.ndarray):
    """Энкодер запроса — обычная функция, ядро не знает, что за ней стоит."""
    return lambda text: vec


@pytest.fixture
def store(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    result = store.add_photos([make_image() for _ in range(SIZE)])
    return store, [p.photo_id for p in result.added]


def caption_all(store, ids, *, matching: str | None = None, match_vector=None):
    """Размечает всю базу: всем случайные векторы, одному — заданный."""
    texts = {pid: f"подпись {i}" for i, pid in enumerate(ids)}
    vectors = {pid: vector(100 + i) for i, pid in enumerate(ids)}
    if matching is not None:
        texts[matching] = "красный автобус"
        vectors[matching] = match_vector
    store.set_caption_texts(texts)
    store.set_caption_vectors(vectors, model="mini")


# --------------------------------------------------------------------------
# Базовое поведение
# --------------------------------------------------------------------------

def test_without_encoder_behaves_as_before(store):
    s, ids = store
    caption_all(s, ids)

    _, plain = s.search_text("кот", top_k=3, translate=False)

    assert len(plain) == 3


def test_caption_lifts_matching_photo(store):
    """
    Снимок, чья подпись точно отвечает запросу, должен заметно подняться. Берём
    тот, который CLIP ставит последним, иначе проверять нечего.

    Первого места здесь не требуется, и это не послабление теста: при весе 0,7
    путь CLIP главнее, поэтому снимок с сильной подписью обязан подняться, но не
    обязан вытеснять то, что CLIP нашёл уверенно. Требование «стал первым»
    означало бы, что подписи перебивают основной путь, — а измеряли не это.
    """
    s, ids = store
    _, baseline = s.search_text("что угодно", top_k=SIZE, translate=False)
    outsider = baseline[-1].photo_id
    caption_all(s, ids, matching=outsider, match_vector=vector(42))

    _, fused = s.search_text(
        "красный автобус", top_k=SIZE, translate=False,
        caption_encoder=encoder_for(vector(42)),
    )

    rank = [hit.photo_id for hit in fused].index(outsider)
    assert rank < SIZE // 2, f"был последним из {SIZE}, стал {rank + 1}-м — не поднялся"
    assert fused[rank].caption == "красный автобус"


def test_hit_carries_caption(store):
    s, ids = store
    caption_all(s, ids)

    _, hits = s.search_text("запрос", top_k=3, translate=False,
                            caption_encoder=encoder_for(vector(1)))

    assert all(hit.caption for hit in hits)


# --------------------------------------------------------------------------
# Неполное покрытие
# --------------------------------------------------------------------------

def test_fusion_off_below_coverage_threshold(store):
    """
    Мало подписей — слияние не включается вовсе, иначе оценка нормируется по
    горстке снимков и лучший из них вылезает наверх без всяких оснований.
    """
    s, ids = store
    few = ids[:3]
    s.set_caption_texts({pid: "подпись" for pid in few})
    s.set_caption_vectors({pid: vector(i) for i, pid in enumerate(few)}, model="mini")

    assert s.fusion_ready() is False

    _, plain = s.search_text("запрос", top_k=SIZE, translate=False)
    _, fused = s.search_text("запрос", top_k=SIZE, translate=False,
                             caption_encoder=encoder_for(vector(0)))

    assert [h.photo_id for h in fused] == [h.photo_id for h in plain]


def test_fusion_on_once_half_the_base_is_captioned(store):
    s, ids = store
    half = ids[: SIZE // 2]
    s.set_caption_texts({pid: "подпись" for pid in half})
    s.set_caption_vectors({pid: vector(i) for i, pid in enumerate(half)}, model="mini")

    assert len(half) >= CAPTION_FUSION_MIN_PHOTOS
    assert len(half) / SIZE >= CAPTION_FUSION_MIN_COVERAGE
    assert s.fusion_ready() is True


def test_uncaptioned_photo_scores_exactly_the_caption_mean(store):
    """
    Снимку без подписи подставляется среднее по подписанным, а после нормировки
    это ровно ноль. Проверяется само значение, а не порядок: равномерный штраф
    всем неразмеченным снимкам порядок между ними сохраняет, поэтому по одному
    лишь порядку такую порчу не увидеть.

    При весе 0 вклад CLIP выключен, и оценка неразмеченного снимка обязана быть
    в точности нулевой.
    """
    s, ids = store
    captioned, bare = ids[: SIZE // 2], ids[SIZE // 2:]
    s.set_caption_texts({pid: "подпись" for pid in captioned})
    s.set_caption_vectors({pid: vector(i) for i, pid in enumerate(captioned)}, model="mini")

    _, hits = s.search_text("запрос", top_k=SIZE, translate=False,
                            caption_encoder=encoder_for(vector(3)), alpha=0.0)

    scores = {hit.photo_id: hit.score for hit in hits}
    for photo_id in bare:
        assert scores[photo_id] == pytest.approx(0.0, abs=1e-6)
    assert any(scores[pid] > 0.1 for pid in captioned)  # у размеченных разброс есть


def test_photo_without_caption_keeps_its_clip_order(store):
    """
    Нейтральность подстановки: снимки без подписи сохраняют взаимный порядок,
    заданный CLIP, а не проваливаются скопом вниз.
    """
    s, ids = store
    captioned, bare = ids[: SIZE // 2], ids[SIZE // 2:]
    s.set_caption_texts({pid: "подпись" for pid in captioned})
    s.set_caption_vectors({pid: vector(i) for i, pid in enumerate(captioned)}, model="mini")

    _, plain = s.search_text("запрос", top_k=SIZE, translate=False)
    _, fused = s.search_text("запрос", top_k=SIZE, translate=False,
                             caption_encoder=encoder_for(vector(3)))

    before = [pid for pid in (h.photo_id for h in plain) if pid in set(bare)]
    after = [pid for pid in (h.photo_id for h in fused) if pid in set(bare)]
    assert after == before


# --------------------------------------------------------------------------
# Вес слияния
# --------------------------------------------------------------------------

def test_default_alpha_matches_measurement():
    """
    Вес не должен разъехаться с отчётом. Значение взято из замера на МАШИННЫХ
    подписях (C3), а не на человеческих (C0): работать система будет на машинных,
    и оптимум у них другой — 0,7 против 0,6.
    """
    assert CAPTION_FUSION_ALPHA == 0.7


def test_alpha_one_reproduces_clip_order(store):
    s, ids = store
    caption_all(s, ids)

    _, plain = s.search_text("запрос", top_k=SIZE, translate=False)
    _, fused = s.search_text("запрос", top_k=SIZE, translate=False,
                             caption_encoder=encoder_for(vector(5)), alpha=1.0)

    assert [h.photo_id for h in fused] == [h.photo_id for h in plain]


def test_alpha_zero_ranks_by_captions_only(store):
    s, ids = store
    target = ids[2]
    caption_all(s, ids, matching=target, match_vector=vector(3))

    _, fused = s.search_text("кот", top_k=SIZE, translate=False,
                             caption_encoder=encoder_for(vector(3)), alpha=0.0)

    assert fused[0].photo_id == target


# --------------------------------------------------------------------------
# Края
# --------------------------------------------------------------------------

def test_scores_do_not_depend_on_requested_top_k(store):
    """
    Нормировка обязана считаться по всей базе. Если рантайм посчитает среднее и
    разброс по верхушке списка, вес 0.6 перестанет означать то, что измерено, —
    и заметно это будет только по тому, что оценки поплывут вслед за top_k.
    """
    s, ids = store
    caption_all(s, ids)

    _, few = s.search_text("запрос", top_k=2, translate=False,
                           caption_encoder=encoder_for(vector(0)))
    _, many = s.search_text("запрос", top_k=SIZE, translate=False,
                            caption_encoder=encoder_for(vector(0)))

    by_id = {hit.photo_id: hit.score for hit in many}
    for hit in few:
        assert hit.score == pytest.approx(by_id[hit.photo_id], abs=1e-6)


def test_deleting_captions_below_threshold_falls_back(store):
    s, ids = store
    caption_all(s, ids)
    assert s.fusion_ready() is True

    s.delete_photos(ids[:SIZE - 3])  # осталось 3 подписи

    assert s.fusion_ready() is False
    _, hits = s.search_text("запрос", top_k=3, translate=False,
                            caption_encoder=encoder_for(vector(1)))
    assert len(hits) == 3


def test_wrong_query_dimension_is_rejected(store):
    s, ids = store
    caption_all(s, ids)

    with pytest.raises(StoreError):
        s.search_text("запрос", top_k=3, translate=False,
                      caption_encoder=encoder_for(vector(1, 16)))
