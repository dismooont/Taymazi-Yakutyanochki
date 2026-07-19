"""
Кэш открытых баз (IndexStore) и синхронизация их статистики с SQLite.

Держать в памяти все базы всех пользователей нельзя: images.index на 5000 фото — 10 МБ,
а captions.index того же датасета — уже 50 МБ. Поэтому LRU на несколько баз с вытеснением
по времени простоя (docs/WEB_PLAN.md, раздел 4.2).

Счётчики объёма живут в БД, а не пересчитываются обходом папки на каждый рендер списка баз.
Единственное место, где они обновляются, — sync_stats(), вызываемая после любой операции,
меняющей базу.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

from core.store import IndexStore
from web import db
from web.config import get_settings

DEFAULT_CAPACITY = 4
DEFAULT_TTL_SECONDS = 15 * 60


class StoreCache:
    def __init__(self, capacity: int = DEFAULT_CAPACITY, ttl: float = DEFAULT_TTL_SECONDS):
        self.capacity = capacity
        self.ttl = ttl
        self._items: OrderedDict[str, tuple[IndexStore, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, user_id: str, database_id: str) -> IndexStore:
        """Открывает базу (или создаёт папку с пустым индексом) и кладёт её в кэш."""
        with self._lock:
            self._drop_stale()
            found = self._items.get(database_id)
            if found is not None:
                store, _ = found
                self._items[database_id] = (store, time.monotonic())
                self._items.move_to_end(database_id)
                return store

        # открытие вне лока: чтение индекса с диска может занять секунды,
        # и на это время не должны блокироваться остальные базы
        root = get_settings().database_dir(user_id, database_id)
        store = IndexStore.open_or_create(root)

        with self._lock:
            self._items[database_id] = (store, time.monotonic())
            self._items.move_to_end(database_id)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return store

    def evict(self, database_id: str) -> None:
        with self._lock:
            self._items.pop(database_id, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _drop_stale(self) -> None:
        deadline = time.monotonic() - self.ttl
        for key in [k for k, (_, seen) in self._items.items() if seen < deadline]:
            self._items.pop(key, None)


store_cache = StoreCache()


PREVIEW_COUNT = 4


def sync_stats(database_id: str, store: IndexStore) -> dict:
    """Переносит реальные показатели базы в SQLite и возвращает обновлённую строку."""
    stats = store.stats()
    # id первых снимков сохраняем здесь же: список баз показывает по ним превью,
    # и открывать ради этого каждую базу с диска было бы расточительно
    preview = [photo.photo_id for photo in store.list_photos(0, PREVIEW_COUNT)]
    db.update_database_stats(
        database_id,
        photos_count=stats.photos_count,
        photos_bytes=stats.photos_bytes,
        index_bytes=stats.index_bytes,
        has_captions=stats.has_captions,
        preview=preview,
    )
    return db.get_database(database_id)


def create_store(user_id: str, database_id: str) -> IndexStore:
    root = get_settings().database_dir(user_id, database_id)
    store = IndexStore.create_empty(root)
    store_cache.evict(database_id)
    return store


def remove_store_files(user_id: str, database_id: str) -> None:
    """
    Удаляет папку базы с диска. Вызывается после удаления строки из БД: если упасть
    между шагами, лучше остаться с осиротевшими файлами, чем с записью в БД без файлов.
    """
    store_cache.evict(database_id)
    root: Path = get_settings().database_dir(user_id, database_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
