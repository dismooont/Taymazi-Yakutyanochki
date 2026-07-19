"""
Общие фикстуры тестов.

Тесты не должны тянуть 600 МБ весов CLIP ради проверки логики индекса, поэтому синглтон
ModelHolder подменяется фейком (ModelHolder.set_instance). Фейк детерминированный: одна
и та же картинка всегда даёт один и тот же вектор — этого достаточно, чтобы проверять
главный инвариант базы (i-я запись meta описывает i-й вектор индекса).

Качество самого CLIP тестами не проверяется — за это отвечает scripts/eval_recall.py.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model import ModelHolder  # noqa: E402

FAKE_DIM = 32


def _vector_from_seed(seed: int, dim: int = FAKE_DIM) -> np.ndarray:
    """Псевдослучайный, но воспроизводимый единичный вектор."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype("float32")
    return vec / np.linalg.norm(vec)


class FakeHolder(ModelHolder):
    """
    Подмена ModelHolder без реальной модели.

    Картинка кодируется по цвету левого верхнего пикселя, текст — по crc32 строки
    (не встроенный hash(): он рандомизируется от запуска к запуску, и тесты стали бы
    невоспроизводимыми).
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._dim = FAKE_DIM
        import threading

        self._lock = threading.Lock()
        self.calls = {"images": 0, "texts": 0}

    @property
    def dim(self) -> int:
        return FAKE_DIM

    def encode_images(self, images):
        self.calls["images"] += len(images)
        out = []
        for im in images:
            r, g, b = im.convert("RGB").getpixel((0, 0))
            out.append(_vector_from_seed(r * 65536 + g * 256 + b))
        return np.ascontiguousarray(np.array(out, dtype="float32").reshape(-1, FAKE_DIM))

    def encode_texts(self, texts):
        self.calls["texts"] += len(texts)
        out = [_vector_from_seed(zlib.crc32(t.encode("utf-8"))) for t in texts]
        return np.ascontiguousarray(np.array(out, dtype="float32").reshape(-1, FAKE_DIM))


@pytest.fixture
def holder():
    fake = FakeHolder()
    ModelHolder.set_instance(fake)
    yield fake
    ModelHolder.set_instance(None)


@pytest.fixture
def make_image(tmp_path):
    """Создаёт JPEG сплошного цвета: цвет -> предсказуемый эмбеддинг у FakeHolder."""
    counter = {"n": 0}

    def _make(color: tuple[int, int, int] | None = None, name: str | None = None,
              suffix: str = ".jpg", size: tuple[int, int] = (64, 64)) -> Path:
        counter["n"] += 1
        n = counter["n"]
        color = color or (n * 7 % 256, n * 13 % 256, n * 29 % 256)
        path = tmp_path / "src" / (name or f"img{n}{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    return _make
