"""
Перевод текстовых запросов RU->EN с файловым кэшем.

CLIP обучен на английских подписях, поэтому русский запрос сначала переводится.
Кэш лежит внутри папки базы (translate_cache.json), чтобы не расходовать лимиты
внешнего API на повторяющиеся запросы.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TRANSLATE_CACHE_FILE = "translate_cache.json"


def has_cyrillic(text: str) -> bool:
    """Грубая эвристика: есть ли в строке кириллица (значит запрос на русском)."""
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def translate_ru_to_en(text: str, cache_path: str | Path) -> str:
    """
    Переводит запрос RU->EN. При любой ошибке переводчика возвращает исходный текст
    и НЕ кэширует неудачу — чтобы следующая попытка прошла заново.
    """
    cache_path = str(cache_path)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}  # битый кэш не должен ронять поиск
    if text in cache:
        return cache[text]

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="ru", target="en").translate(text)
        if not translated:
            raise ValueError("Переводчик вернул пустую строку")
    except Exception as e:
        print(f"[перевод не удался, используется исходный текст без кэширования] {e}")
        return text

    cache[text] = translated
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return translated


def maybe_translate(query: str, cache_path: str | Path, enabled: bool = True) -> str:
    """Переводит только если включено и в строке действительно есть кириллица."""
    if enabled and has_cyrillic(query):
        return translate_ru_to_en(query, cache_path)
    return query
