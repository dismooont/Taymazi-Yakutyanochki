"""
Текстовый энкодер для поиска по подписям (фаза C2).

Зачем отдельно от core/model.py: CLIP нужен всегда, а этот — только если у базы
есть подписи. Модель грузится лениво и ровно один раз на процесс, как и CLIP,
но её отсутствие ничего не ломает: поиск просто остаётся обычным.

sentence-transformers импортируется внутри метода, а не на уровне модуля. Иначе
`import core` тянул бы за собой ещё одну модель, а бот, который вообще не считает
эмбеддинги, не должен даже знать о ней.
"""

from __future__ import annotations

import threading

import numpy as np

DEFAULT_CAPTION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CaptionEncoder:
    """
    Обёртка над текстовой моделью. Потокобезопасна: инференс идёт под локом, как
    и в ModelHolder, — uvicorn отдаёт запросы из пула потоков, а torch на CPU
    параллельные прогоны одной модели не любит.
    """

    _instance: "CaptionEncoder | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_CAPTION_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str = DEFAULT_CAPTION_MODEL) -> "CaptionEncoder":
        if cls._instance is None or cls._instance.model_name != model_name:
            with cls._instance_lock:
                if cls._instance is None or cls._instance.model_name != model_name:
                    cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "CaptionEncoder | None") -> None:
        """Подмена в тестах — веса в тестах не грузятся."""
        cls._instance = instance

    @property
    def dim(self) -> int:
        model = self._load()
        # Метод переименован в новых версиях sentence-transformers; старое имя ещё
        # работает, но шумит FutureWarning. Берём новое, откатываясь на старое.
        getter = getattr(model, "get_embedding_dimension", None) or \
            model.get_sentence_embedding_dimension
        return int(getter())

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    print(f"Загрузка текстовой модели {self.model_name}...")
                    self._model = SentenceTransformer(self.model_name)
                    print(f"Текстовая модель готова: dim={self.dim}")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Нормированные векторы — поиск идёт скалярным произведением в IndexFlatIP."""
        model = self._load()
        with self._lock:
            vectors = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.ascontiguousarray(vectors, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def caption_encoder_available() -> bool:
    """Установлен ли sentence-transformers. Веса при этой проверке не грузятся."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False
