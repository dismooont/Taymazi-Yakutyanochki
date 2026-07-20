"""
Тесты перевода запросов RU->EN.

В сеть тесты не ходят: внешний переводчик подменяется. Проверяется поведение
вокруг него — кэш и, главное, что его недоступность не ломает поиск.
"""

from __future__ import annotations

import builtins
import json
import sys
import types

import pytest

from core import translate as tr


@pytest.fixture
def fake_translator(monkeypatch):
    """Подменяет deep_translator: настоящий перевод требует сети."""
    module = types.ModuleType("deep_translator")

    class GoogleTranslator:
        def __init__(self, source="ru", target="en"):
            pass

        def translate(self, text):
            return f"[EN] {text}"

    module.GoogleTranslator = GoogleTranslator
    monkeypatch.setitem(sys.modules, "deep_translator", module)
    return module


def test_reads_from_cache_without_translator(tmp_path):
    """Готовый перевод берётся из кэша — переводчик не нужен вовсе."""
    cache = tmp_path / "translate_cache.json"
    cache.write_text(json.dumps({"рыжий кот": "red cat"}), encoding="utf-8")

    assert tr.translate_ru_to_en("рыжий кот", cache) == "red cat"


def test_translation_is_cached(tmp_path, fake_translator):
    cache = tmp_path / "translate_cache.json"

    assert tr.translate_ru_to_en("рыжий кот", cache) == "[EN] рыжий кот"

    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["рыжий кот"] == "[EN] рыжий кот"


def test_search_survives_read_only_cache(tmp_path, fake_translator, monkeypatch):
    """
    Демо-база примонтирована только для чтения, и запись кэша в неё падает
    с OSError. На живом стенде это роняло каждый русский запрос к демо-базе
    с ошибкой 500. Перевод при этом уже получен — его и надо вернуть.
    """
    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode:
            raise OSError(30, "Read-only file system")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only_open)

    result = tr.translate_ru_to_en("рыжий кот", tmp_path / "translate_cache.json")

    assert result == "[EN] рыжий кот"


def test_failed_translation_is_not_cached(tmp_path, monkeypatch):
    """Неудачный перевод кэшировать нельзя — иначе ошибка закрепится навсегда."""
    module = types.ModuleType("deep_translator")

    class Broken:
        def __init__(self, **kwargs):
            pass

        def translate(self, text):
            raise RuntimeError("сеть недоступна")

    module.GoogleTranslator = Broken
    monkeypatch.setitem(sys.modules, "deep_translator", module)

    cache = tmp_path / "translate_cache.json"
    result = tr.translate_ru_to_en("рыжий кот", cache)

    assert result == "рыжий кот"  # вернулся исходный текст
    assert not cache.exists()


def test_maybe_translate_skips_latin(tmp_path, fake_translator):
    """Английский запрос переводить незачем."""
    assert tr.maybe_translate("red cat", tmp_path / "c.json", enabled=True) == "red cat"


def test_maybe_translate_respects_flag(tmp_path, fake_translator):
    assert tr.maybe_translate("рыжий кот", tmp_path / "c.json", enabled=False) == "рыжий кот"
