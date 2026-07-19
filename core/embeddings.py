"""
Пакетный подсчёт эмбеддингов поверх ModelHolder.

Отличие от прежнего кода: появился on_progress — колбэк, через который фоновая задача
веб-приложения отдаёт прогресс индексации в UI (docs/WEB_PLAN.md, раздел 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from core.model import BATCH_SIZE, ModelHolder

ProgressCallback = Callable[[int, int], None]


def _iter_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def open_image(path: str | Path) -> Image.Image:
    """
    Открывает картинку в том виде, в каком её ждёт CLIP: RGB и с учётом EXIF-поворота.
    Без exif_transpose часть фото с телефонов уходит в модель лежащей на боку.
    """
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def compute_image_embeddings(
    image_paths: Sequence[str | Path],
    *,
    batch_size: int = BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
    show_progress: bool = True,
    holder: ModelHolder | None = None,
) -> np.ndarray:
    """Эмбеддинги списка файлов: (len(image_paths), dim), нормализованные."""
    holder = holder or ModelHolder.get()
    total = len(image_paths)
    if total == 0:
        return np.zeros((0, holder.dim), dtype="float32")

    chunks, done = [], 0
    batches = list(_iter_batches(image_paths, batch_size))
    for batch in tqdm(batches, desc="Эмбеддинги изображений", disable=not show_progress):
        images = [open_image(p) for p in batch]
        chunks.append(holder.encode_images(images))
        done += len(batch)
        if on_progress is not None:
            on_progress(done, total)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype="float32")


def compute_text_embeddings(
    texts: Sequence[str],
    *,
    batch_size: int = BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
    show_progress: bool = True,
    holder: ModelHolder | None = None,
) -> np.ndarray:
    """Эмбеддинги списка текстов: (len(texts), dim), нормализованные."""
    holder = holder or ModelHolder.get()
    total = len(texts)
    if total == 0:
        return np.zeros((0, holder.dim), dtype="float32")

    chunks, done = [], 0
    batches = list(_iter_batches(list(texts), batch_size))
    for batch in tqdm(batches, desc="Эмбеддинги текстов", disable=not show_progress):
        chunks.append(holder.encode_texts(batch))
        done += len(batch)
        if on_progress is not None:
            on_progress(done, total)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype="float32")
