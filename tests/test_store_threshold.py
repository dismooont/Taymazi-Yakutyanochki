"""
Тесты порога «показывать только похожее» (core).

У заглушки модели картинка кодируется по цвету левого верхнего пикселя, поэтому
косинусы управляемы: запрос-картинка того же цвета, что снимок в базе, даёт с ним
косинус 1.0, а с остальными — около нуля. На этом и строится проверка: высокий
порог оставляет только точное совпадение, отсутствие порога — весь top_k.

Абсолютные значения (0,24 / 0,75) здесь не проверяются — они измерены на реальном
CLIP (см. core/store.py). Здесь проверяется механизм отсечения.
"""

from __future__ import annotations

import pytest

from core.store import DEFAULT_IMAGE_MIN_SCORE, DEFAULT_TEXT_MIN_SCORE, IndexStore


@pytest.fixture
def store(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    # три снимка разных цветов -> три далёких эмбеддинга
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
    files = [make_image(color=c) for c in colors]
    result = store.add_photos(files)
    return store, colors, [p.photo_id for p in result.added], make_image


def test_defaults_are_the_measured_values():
    """Пороги-умолчания не должны разъехаться с замером в core/store.py."""
    assert DEFAULT_TEXT_MIN_SCORE == 0.24
    assert DEFAULT_IMAGE_MIN_SCORE == 0.75


def test_no_threshold_returns_everything(store):
    s, colors, ids, make_image = store
    query = make_image(color=colors[0])

    hits, _ = s.search_image(query, top_k=10)  # min_score по умолчанию None

    assert len(hits) == 3  # весь top_k, как раньше


def test_high_threshold_keeps_only_the_match(store):
    """
    Ровно пользовательский сценарий: запрос точно похож на один снимок, остальные
    далеки. Высокий порог оставляет только похожий, а не добирает выдачу «чем есть».
    """
    s, colors, ids, make_image = store
    query = make_image(color=colors[1])  # точный цвет второго снимка -> косинус 1.0

    hits, _ = s.search_image(query, top_k=10, min_score=0.9)

    assert len(hits) == 1
    assert hits[0].photo_id == ids[1]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_impossible_threshold_returns_empty(store):
    """Порог выше единицы недостижим для косинуса — выдача пуста, а не падает."""
    s, colors, ids, make_image = store
    query = make_image(color=colors[0])

    hits, _ = s.search_image(query, top_k=10, min_score=1.5)

    assert hits == []


def test_text_threshold_filters(store):
    s, _, _, _ = store
    _, high = s.search_text("что-нибудь", top_k=10, translate=False, min_score=1.5)
    _, none = s.search_text("что-нибудь", top_k=10, translate=False)

    assert high == []          # ничего не дотянуло до недостижимого порога
    assert len(none) == 3      # без порога — весь top_k


def test_search_similar_respects_threshold(store):
    s, colors, ids, make_image = store
    # похожие на первый снимок: сам он исключён, остальные далеки (косинус ~0)
    assert s.search_similar(ids[0], top_k=10, min_score=0.9) == []
    assert len(s.search_similar(ids[0], top_k=10)) >= 1  # без порога кто-то да найдётся


def test_captions_are_not_filtered_by_image_threshold(tmp_path, holder, make_image):
    """
    Порог фото не должен трогать ближайшие подписи: это отдельная величина. Проверяем
    на демо-подобной базе с индексом подписей нельзя без COCO, поэтому здесь только
    убеждаемся, что у базы без подписей высокий порог не роняет search_image.
    """
    store = IndexStore.create_empty(tmp_path / "db")
    store.add_photos([make_image() for _ in range(3)])

    hits, captions = store.search_image(make_image(), top_k=5, min_score=0.9)

    assert captions == []  # подписей нет — но и не упали
