"""
Генератор подписей к снимкам (фаза C3).

Новых зависимостей не требует: BLIP лежит в transformers, который уже стоит ради
CLIP. Цена — только веса, около 990 МБ, и они качаются при первом обращении.

Модель грузится лениво и ровно один раз на процесс, как CLIP и текстовый энкодер.
Отдельно от них потому, что нужна реже всех: подписи генерируются пакетно и в
фоне, а поиск работает и без них.
"""

from __future__ import annotations

import threading
from pathlib import Path

from core.embeddings import open_image

DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
# Веса можно положить рядом с проектом и не ходить в сеть. Это не только про
# офлайн: huggingface_hub на большом файле здесь молча стоит на нуле, а обычная
# докачка curl идёт — так что локальная папка оказывается ещё и надёжнее.
LOCAL_BLIP_DIR = Path(__file__).resolve().parent.parent / "models" / "blip"
# Подпись к фотографии — это одно короткое предложение.
MAX_NEW_TOKENS = 24
# Жадная генерация на однообразных снимках срывается в повтор: офисный кубикл дал
# «a cubic cubic cubic cubic ...» на все двадцать четыре токена. Ограничение длины
# от этого не спасает, оно лишь обрезает повтор — нужен запрет на повторение пар
# слов. Такая подпись не просто бесполезна: она попадает в индекс и притягивает к
# себе запросы со словом из повтора.
NO_REPEAT_NGRAM = 2
REPETITION_PENALTY = 1.2


class Captioner:
    """
    Обёртка над BLIP. Инференс под локом — как в ModelHolder: torch на CPU не любит
    параллельные прогоны одной модели.
    """

    _instance: "Captioner | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_BLIP_MODEL, num_threads: int | None = None):
        self.model_name = model_name
        self.num_threads = num_threads
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str = DEFAULT_BLIP_MODEL, num_threads: int | None = None):
        if cls._instance is None or cls._instance.model_name != model_name:
            with cls._instance_lock:
                if cls._instance is None or cls._instance.model_name != model_name:
                    cls._instance = cls(model_name, num_threads)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "Captioner | None") -> None:
        """Подмена в тестах — веса в тестах не грузятся."""
        cls._instance = instance

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from transformers import BlipForConditionalGeneration, BlipProcessor

                    if self.num_threads:
                        # Фоновая генерация не должна отбирать все ядра у поиска:
                        # torch по умолчанию занимает их столько, сколько найдёт.
                        torch.set_num_threads(self.num_threads)

                    source = self.model_name
                    if (LOCAL_BLIP_DIR / "pytorch_model.bin").exists():
                        source = str(LOCAL_BLIP_DIR)

                    print(f"Загрузка BLIP {source}...")
                    self._processor = BlipProcessor.from_pretrained(source)
                    self._model = BlipForConditionalGeneration.from_pretrained(source)
                    self._model.eval()
                    print("BLIP готов")
        return self._model, self._processor

    def caption_images(self, paths: list[str | Path | Image.Image], batch_size: int = 4) -> list[str]:
        """
        Подписи к файлам (или уже открытым картинкам), по одной, в том же порядке.

        Нечитаемый файл даёт пустую подпись, а не исключение: разметка идёт пакетно
        по всей базе, и один битый снимок не должен отменять остальные. Принимает и
        готовый PIL.Image — нужно для подписи картинки-запроса поиска по образцу
        (web/routers/search.py): она уже в памяти, сохранять на диск незачем.
        """
        import torch
        from PIL import Image

        model, processor = self._load()
        captions: list[str] = []

        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            images, positions = [], []
            for offset, item in enumerate(chunk):
                try:
                    images.append(item if isinstance(item, Image.Image) else open_image(item))
                    positions.append(offset)
                except Exception as e:  # noqa: BLE001 — причина уходит в лог, снимок пропускаем
                    print(f"[подпись не создана] {item}: {e}")

            produced = [""] * len(chunk)
            if images:
                with self._lock, torch.inference_mode():
                    inputs = processor(images=images, return_tensors="pt")
                    output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM,
                        repetition_penalty=REPETITION_PENALTY,
                    )
                    decoded = processor.batch_decode(output, skip_special_tokens=True)
                for offset, text in zip(positions, decoded):
                    produced[offset] = text.strip()
            captions.extend(produced)

        return captions

    def caption_one(self, path: str | Path | Image.Image) -> str:
        return self.caption_images([path])[0]
