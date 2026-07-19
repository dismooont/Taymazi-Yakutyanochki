"""
Резидентная модель CLIP: одна загрузка весов на процесс.

Причина существования этого модуля (см. docs/WEB_PLAN.md, раздел 2): веса CLIP занимают
~600 МБ RAM, поэтому веб-приложение, бот и CLI не должны загружать их повторно. ModelHolder —
синглтон: первый вызов get() грузит модель, все последующие возвращают тот же объект.

Инференс защищён блокировкой: в вебе поиск и фоновая индексация идут из разных потоков
(asyncio.to_thread), а один и тот же CLIPProcessor/CLIPModel параллельно дёргать нельзя.
"""

from __future__ import annotations

import threading

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


def extract_features_tensor(output):
    """
    Совместимость между версиями transformers:
    - в transformers < 5.0 get_image_features/get_text_features возвращают torch.Tensor напрямую
    - в transformers >= 5.0 возвращается объект BaseModelOutputWithPooling
    Извлекаем сам тензор эмбеддинга независимо от версии.
    """
    if torch.is_tensor(output):
        return output
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        if hasattr(output, attr):
            tensor = getattr(output, attr)
            if tensor is not None:
                # last_hidden_state имеет форму (batch, seq_len, dim) -> берём CLS-токен
                if attr == "last_hidden_state" and tensor.dim() == 3:
                    tensor = tensor[:, 0, :]
                return tensor
    raise TypeError(f"Не удалось извлечь тензор эмбеддинга из объекта типа {type(output)}")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Нормализация к единичной длине + приведение к тому виду, который ждёт FAISS:
    float32, C-contiguous. IndexFlatIP на нормализованных векторах даёт косинусную близость.
    """
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return np.ascontiguousarray(vectors / norms, dtype="float32")


def _load_pretrained():
    """
    Грузит веса, переживая недоступность huggingface.co.

    transformers при каждом from_pretrained ходит в сеть проверить, не обновилась ли модель,
    и падает, если провайдер рвёт SSL или сервер стоит за прокси, — даже когда веса уже
    лежат в локальном кэше. Поэтому при сетевой ошибке повторяем с local_files_only=True.
    Веб-приложение и бот должны стартовать без интернета.
    """
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        return model, processor
    except Exception as e:
        if not _looks_like_network_error(e):
            raise
        print(f"[сеть недоступна: {type(e).__name__}] — беру модель из локального кэша")
        try:
            model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True).to(DEVICE).eval()
            processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
            return model, processor
        except Exception as offline_error:
            raise RuntimeError(
                f"Не удалось загрузить {MODEL_NAME}: сеть недоступна, а в локальном кэше "
                f"весов нет. Скачайте модель при работающем интернете хотя бы один раз."
            ) from offline_error


def _looks_like_network_error(exc: BaseException) -> bool:
    """Отличает сетевой сбой от настоящей ошибки загрузки (битые веса, нет такой модели)."""
    markers = ("SSL", "Connection", "Timeout", "MaxRetry", "Network", "Proxy", "Temporarily")
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(m.lower() in type(exc).__name__.lower() for m in markers):
            return True
        if any(m.lower() in str(exc).lower() for m in markers):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class ModelHolder:
    """
    Держит CLIP и процессор в памяти. Единственное место в проекте, которое обращается
    к модели напрямую, — поэтому здесь же живёт блокировка на время инференса.

    Использование:
        holder = ModelHolder.get()
        emb = holder.encode_texts(["a dog in the snow"])   # (1, dim), нормализован
    """

    _instance: "ModelHolder | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model, processor, device: str = DEVICE):
        self.model = model
        self.processor = processor
        self.device = device
        self._lock = threading.Lock()
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "ModelHolder":
        """Возвращает синглтон, загружая модель при первом обращении (double-checked locking)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    print(f"Загрузка модели {MODEL_NAME} на {DEVICE}...")
                    model, processor = _load_pretrained()
                    cls._instance = cls(model, processor)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "ModelHolder | None") -> None:
        """
        Подменяет синглтон. Нужно тестам, чтобы не тянуть 600 МБ весов ради проверки
        логики индекса (см. tests/conftest.py). В обычном коде вызывать не следует.
        """
        cls._instance = instance

    @property
    def dim(self) -> int:
        """Размерность эмбеддинга. Берётся из конфига модели, а не хардкодится."""
        if self._dim is None:
            configured = getattr(self.model.config, "projection_dim", None)
            # у части версий transformers поля нет — тогда определяем пробным прогоном
            self._dim = int(configured) if configured else int(self.encode_texts(["x"]).shape[1])
        return self._dim

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Эмбеддинги пачки изображений: (len(images), dim), нормализованные."""
        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        with self._lock:
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                feats = extract_features_tensor(self.model.get_image_features(**inputs))
                return l2_normalize(feats.cpu().numpy())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Эмбеддинги пачки текстов: (len(texts), dim), нормализованные."""
        if not texts:
            # без обращения к self.dim: он сам может пробовать определять размерность
            # через encode_texts(["x"]) и получилась бы рекурсия
            return np.zeros((0, self._dim or 0), dtype="float32")
        with self._lock:
            with torch.no_grad():
                inputs = self.processor(
                    text=texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                feats = extract_features_tensor(self.model.get_text_features(**inputs))
                return l2_normalize(feats.cpu().numpy())


def load_model():
    """
    Обратная совместимость с CLI и ботом: раньше load_model() возвращал (model, processor).
    Теперь это те же объекты из синглтона, то есть повторный вызов не грузит вторую копию.
    """
    holder = ModelHolder.get()
    return holder.model, holder.processor
