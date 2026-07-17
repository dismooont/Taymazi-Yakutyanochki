"""
Резидентная обёртка над CLIP + FAISS для Telegram-бота.

В отличие от CLI-скрипта (src/clip_zero_shot_search.py), который грузит модель
и индексы заново на каждый вызов процесса, класс SearchEngine загружает модель,
процессор и оба FAISS-индекса ОДИН РАЗ при старте и переиспользует их на каждое
сообщение пользователя (см. README, раздел 9.1).

Низкоуровневые функции (эмбеддинги, перевод, разбор image_id) переиспользуются
из src/clip_zero_shot_search.py, чтобы не дублировать логику и держать поведение
бота идентичным CLI.
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

# --- подключаем src/ как модуль, переиспользуем его функции ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clip_zero_shot_search as czss  # noqa: E402


def has_cyrillic(text: str) -> bool:
    """Грубая эвристика: есть ли в строке кириллица (значит запрос на русском)."""
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


class SearchEngine:
    """
    Держит модель CLIP, процессор и оба FAISS-индекса резидентно в памяти.

    Использование:
        engine = SearchEngine("index")
        used_query, results = engine.search_by_text("собака в снегу")
        similar, captions = engine.search_by_image(pil_image)
        norm_id, added = engine.add_image("data/user_photos/123.jpg")
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        if not (self.index_dir / "images.index").exists():
            raise FileNotFoundError(
                f"В {self.index_dir} нет построенного индекса (images.index). "
                f"Сначала выполните команду 'build' из src/clip_zero_shot_search.py."
            )

        self.translate_cache_path = str(self.index_dir / czss.TRANSLATE_CACHE_FILE)

        # модель и процессор — грузятся один раз
        self.model, self.processor = czss.load_model()

        # индекс изображений (обязателен)
        self.images_index, self.images_meta = czss.load_index(str(self.index_dir), "images")
        self.existing_ids = {czss.normalize_id(item["image_id"]) for item in self.images_meta}

        # индекс подписей (может отсутствовать, если строили без CSV)
        captions_path = self.index_dir / "captions.index"
        if captions_path.exists():
            self.captions_index, self.captions_meta = czss.load_index(str(self.index_dir), "captions")
        else:
            self.captions_index, self.captions_meta = None, []

        print(
            f"SearchEngine готов: {len(self.images_meta)} изображений, "
            f"{len(self.captions_meta)} подписей, device={czss.DEVICE}"
        )

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search_by_text(self, query: str, top_k: int = 5, translate: bool = True):
        """
        Поиск изображений по тексту. Если translate=True и запрос содержит
        кириллицу — переводит RU->EN перед поиском (CLIP обучен на английском).

        Возвращает (used_query, results), где results — список dict
        {image_id, score, path}, отсортированный по убыванию score.
        """
        used_query = query
        if translate and has_cyrillic(query):
            used_query = czss.translate_ru_to_en(query, self.translate_cache_path)

        emb = czss.compute_text_embeddings(self.model, self.processor, [used_query])
        faiss.normalize_L2(emb)
        scores, indices = self.images_index.search(emb, top_k)
        return used_query, self._collect(self.images_meta, indices[0], scores[0])

    def search_by_image(self, image: Image.Image, top_k: int = 5):
        """
        Поиск по картинке-запросу. Возвращает (similar_images, captions):
        - similar_images: список dict {image_id, score, path}
        - captions: список dict {image_id, score, caption} (пусто, если индекса подписей нет)
        """
        emb = self._embed_image(image)

        scores, indices = self.images_index.search(emb, top_k)
        similar = self._collect(self.images_meta, indices[0], scores[0])

        captions = []
        if self.captions_index is not None:
            cscores, cindices = self.captions_index.search(emb, top_k)
            for idx, score in zip(cindices[0], cscores[0]):
                item = self.captions_meta[idx]
                captions.append(
                    {"image_id": item["image_id"], "score": float(score), "caption": item["caption"]}
                )
        return similar, captions

    # ------------------------------------------------------------------
    # Инкрементальное добавление (сценарий "пользователь прислал фото")
    # ------------------------------------------------------------------

    def add_image(self, image_path: str, image_id: str = None):
        """
        Добавляет одно изображение в индекс изображений (режим 1 из README 6.1:
        без подписи). Модель уже загружена в памяти — повторной загрузки весов нет.

        Возвращает (norm_id, added): added=False, если такой image_id уже в индексе.
        Изменённый индекс сразу сохраняется на диск (переживает перезапуск контейнера).
        """
        if image_id is None:
            image_id = Path(image_path).stem
        norm_id = czss.normalize_id(image_id)
        if norm_id in self.existing_ids:
            return norm_id, False

        image = Image.open(image_path).convert("RGB")
        emb = self._embed_image(image)

        self.images_index.add(emb)
        self.images_meta.append({"image_id": norm_id, "path": str(image_path)})
        self.existing_ids.add(norm_id)

        faiss.write_index(self.images_index, str(self.index_dir / "images.index"))
        with open(self.index_dir / "images_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.images_meta, f, ensure_ascii=False, indent=2)

        return norm_id, True

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Эмбеддинг одной картинки, нормализованный, готовый для FAISS (IndexFlatIP)."""
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt").to(czss.DEVICE)
            feats = czss.extract_features_tensor(self.model.get_image_features(**inputs))
            emb = feats.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb

    @staticmethod
    def _portable_path(raw: str) -> str:
        """
        Пути в meta сохранены как относительные и в стиле ОС, где строился индекс
        (напр. Windows: 'data\\images\\...'). Приводим разделители к '/' и резолвим
        относительно корня проекта, чтобы работало и на Linux в Docker.
        """
        p = raw.replace("\\", "/")
        return p if os.path.isabs(p) else str(PROJECT_ROOT / p)

    def _collect(self, meta, indices, scores):
        results = []
        for idx, score in zip(indices, scores):
            item = meta[idx]
            results.append(
                {
                    "image_id": item["image_id"],
                    "score": float(score),
                    "path": self._portable_path(item["path"]),
                }
            )
        return results
