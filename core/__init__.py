"""
Ядро проекта: модель CLIP, эмбеддинги, перевод запросов и работа с базой (FAISS-индекс).

Этим пакетом одинаково пользуются CLI (src/clip_zero_shot_search.py), Telegram-бот
и веб-приложение — чтобы поведение поиска везде было одинаковым, а веса модели
загружались один раз на процесс.
"""

from core.embeddings import compute_image_embeddings, compute_text_embeddings, open_image
from core.model import BATCH_SIZE, DEVICE, MODEL_NAME, ModelHolder, load_model
from core.store import (
    AddResult,
    CaptionHit,
    IndexStore,
    Photo,
    SearchHit,
    Stats,
    StoreError,
    normalize_id,
)
from core.translate import TRANSLATE_CACHE_FILE, has_cyrillic, maybe_translate, translate_ru_to_en

__all__ = [
    "AddResult",
    "BATCH_SIZE",
    "CaptionHit",
    "DEVICE",
    "IndexStore",
    "MODEL_NAME",
    "ModelHolder",
    "Photo",
    "SearchHit",
    "Stats",
    "StoreError",
    "TRANSLATE_CACHE_FILE",
    "compute_image_embeddings",
    "compute_text_embeddings",
    "has_cyrillic",
    "load_model",
    "maybe_translate",
    "normalize_id",
    "open_image",
    "translate_ru_to_en",
]
