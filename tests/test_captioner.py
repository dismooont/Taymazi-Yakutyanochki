"""
Тесты генератора подписей (фаза C3).

Веса BLIP здесь не грузятся — это почти гигабайт и минуты на прогон. Проверяется
обвязка вокруг модели, и в ней есть что проверять: разметка идёт часами по всей
базе, поэтому важно, что один нечитаемый файл не отменяет остальные, что порядок
подписей совпадает с порядком снимков и что прерывание не теряет сделанное.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.captioner import Captioner
from core.store import IndexStore


class FakeCaptioner(Captioner):
    """Вместо BLIP — имя файла. Порядок и пропуски видны, весов не нужно."""

    def __init__(self, broken: set[str] | None = None):
        super().__init__(model_name="fake")
        self.broken = broken or set()
        self.calls: list[list[str]] = []

    def caption_images(self, paths, batch_size: int = 4) -> list[str]:
        self.calls.append([Path(p).name for p in paths])
        return [
            "" if Path(p).name in self.broken else f"подпись к {Path(p).stem[:6]}"
            for p in paths
        ]


# Больше порога CAPTION_FUSION_MIN_PHOTOS: иначе слияние не включится и проверить
# связку «разметили -> закодировали -> заработало» будет нечем.
SIZE = 12


@pytest.fixture
def base(tmp_path, holder, make_image):
    store = IndexStore.create_empty(tmp_path / "db")
    result = store.add_photos([make_image() for _ in range(SIZE)])
    return store, [p.photo_id for p in result.added]


def run_over(store, captioner, chunk: int = 3, model: str = "fake") -> int:
    """Повторяет то, что делает scripts/generate_captions.py, без разбора аргументов."""
    pending = store.photos_without_caption()
    written = 0
    for start in range(0, len(pending), chunk):
        part = pending[start:start + chunk]
        texts = captioner.caption_images([str(store.photo_path(p)) for p in part])
        written += store.set_caption_texts(
            {photo.photo_id: text for photo, text in zip(part, texts) if text}, model=model
        )
    return written


def test_captions_every_photo(base):
    store, ids = base

    assert run_over(store, FakeCaptioner()) == SIZE
    assert store.captions_coverage() == (SIZE, SIZE)


def test_broken_file_does_not_stop_the_rest(base):
    """Разметка идёт по всей базе часами — один битый снимок не повод её отменять."""
    store, ids = base
    broken = store.photo_path(ids[2]).name

    written = run_over(store, FakeCaptioner(broken={broken}))

    assert written == SIZE - 1
    assert store.caption_of(ids[2]) == ""
    assert store.captions_coverage() == (SIZE - 1, SIZE)


def test_captions_match_their_photos(base):
    """
    Порядок подписей обязан совпадать с порядком снимков. Ошибка здесь тихая:
    подписи просто окажутся не у тех фотографий.
    """
    store, ids = base
    run_over(store, FakeCaptioner())

    for photo in store.list_photos():
        assert photo.caption == f"подпись к {photo.filename[:6]}"


def test_progress_survives_interruption(base):
    """Прерванный прогон должен оставить размеченное на диске, а не потерять его."""
    store, ids = base
    captioner = FakeCaptioner()

    pending = store.photos_without_caption()[:3]
    texts = captioner.caption_images([str(store.photo_path(p)) for p in pending])
    store.set_caption_texts(
        {p.photo_id: t for p, t in zip(pending, texts)}, model="fake"
    )

    reopened = IndexStore.open(store.root)
    assert reopened.captions_coverage() == (3, SIZE)
    assert len(reopened.photos_without_caption()) == SIZE - 3


def test_second_run_only_takes_the_remainder(base):
    """Повторный запуск не должен переразмечать уже размеченное."""
    store, ids = base
    store.set_caption_texts({ids[0]: "уже есть", ids[1]: "и тут"}, model="ручная")

    captioner = FakeCaptioner()
    run_over(store, captioner)

    touched = [name for call in captioner.calls for name in call]
    assert len(touched) == SIZE - 2
    assert store.caption_of(ids[0]) == "уже есть"  # чужая подпись не затёрта


def test_vectors_follow_texts(base):
    """
    После разметки подписи надо закодировать — иначе искать по ним нечем, и
    покрытие в статистике есть, а слияние не включается.
    """
    store, ids = base
    run_over(store, FakeCaptioner())

    photos = [p for p in store.list_photos() if p.caption]
    rng = np.random.default_rng(0)
    vectors = {}
    for photo in photos:
        v = rng.normal(size=8).astype("float32")
        vectors[photo.photo_id] = v / np.linalg.norm(v)
    store.set_caption_vectors(vectors, model="fake-encoder")

    assert store.fusion_ready() is True
    assert store.caption_index_model() == "fake-encoder"
