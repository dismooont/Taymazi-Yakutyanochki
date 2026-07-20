"""
Фоновая разметка баз подписями (фаза C4).

Задача: BLIP считает подпись около полутора секунд на снимок, а поиск отвечает за
десятки миллисекунд. Делать такую работу, пока человек пользуется сайтом, нельзя —
он получит ответ через секунду вместо мгновенного и не поймёт, почему.

Отсюда правило: размечаем только тогда, когда никто ничего не просил уже CAPTION_IDLE
секунд, и прекращаем на первом же запросе.

Почему отдельный поток, а не задача в JobQueue. Очередь задач последовательная и
рассчитана на работу с началом и концом: поставили импорт — он выполнился. Разметка
же длится часами и не имеет конца, поэтому в очереди она заблокировала бы импорты
всех пользователей. Зато оба потока не должны считать одновременно, и это здесь
обеспечено явно: пока в очереди есть активная задача, разметка не начинается.

Три вещи, без которых замысел «работать в простое» не работает:

  * Голодание. Пользователь, который что-то делает раз в двадцать секунд, не даёт
    простою наступить никогда, и подписи не появятся вообще. Поэтому раз в
    CAPTION_FORCE_AFTER секунд разметка идёт независимо от активности — короткими
    отрезками, чтобы это осталось незаметным.
  * Ядра. Прерывание между снимками не спасает, если снимок уже считается: torch по
    умолчанию занимает все ядра, и поиск в этот момент всё равно подтормаживает.
    Ограничение числа потоков делает деградацию предсказуемой.
  * Прогресс на диск. Разметка идёт часами, и перезапуск не должен её обнулять:
    подписи сохраняются после каждого отрезка.
"""

from __future__ import annotations

import threading
import time

from core.captioner import Captioner
from core.captions import CaptionEncoder, caption_encoder_available
from core.store import IndexStore, StoreError
from web import db
from web.config import get_settings
from web.stores import store_for, sync_stats

# Как часто просыпаться и проверять, можно ли работать.
TICK_SECONDS = 5.0
# Сколько снимков размечать за один отрезок, прежде чем снова проверить активность.
# Восемь — это размер пачки BLIP, то есть примерно 13 секунд работы: реже проверять
# значит дольше держать человека в ожидании, чаще — терять на перезапуске пачки.
SLICE_PHOTOS = 8


class Activity:
    """
    Когда последний раз приходил запрос от человека. Общая на приложение.

    Считается по запросам к API, а не по сессиям: важен не факт, что кто-то залогинен,
    а то, что прямо сейчас кто-то ждёт ответа.
    """

    def __init__(self) -> None:
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._last


activity = Activity()


class CaptionWorker:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._last_run = time.monotonic()
        self.captioned_total = 0  # для тестов и лога

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    def start(self) -> None:
        settings = get_settings()
        if not settings.caption_auto_enabled:
            return
        if not caption_encoder_available():
            print("[фоновая разметка выключена] нет sentence-transformers")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="caption-worker", daemon=True)
        self._thread.start()
        print(f"Фоновая разметка включена: простой {settings.caption_idle_seconds} с, "
              f"принудительно раз в {settings.caption_force_after} с")

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------
    # Решение «можно ли сейчас работать»
    # ------------------------------------------------------------------

    def may_work(self) -> bool:
        """
        Пользовательская задача важнее всегда: она с прогрессом на экране и человек
        её ждёт. Простой же можно и пересидеть — но не бесконечно, иначе активный
        пользователь не даст разметить свою базу никогда.
        """
        settings = get_settings()
        if db.has_any_active_job():
            return False
        if activity.idle_seconds() >= settings.caption_idle_seconds:
            return True
        return time.monotonic() - self._last_run >= settings.caption_force_after

    # ------------------------------------------------------------------
    # Работа
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stopping.wait(TICK_SECONDS):
            try:
                if self.may_work():
                    self.run_slice()
            except Exception as e:  # noqa: BLE001 — фоновая работа не имеет права
                # ронять приложение; причина уходит в лог, цикл продолжается
                print(f"[фоновая разметка] {type(e).__name__}: {e}")

    def run_slice(self) -> int:
        """
        Один отрезок работы: найти базу, где не хватает подписей, и разметить
        несколько снимков. Возвращает, сколько разметил.
        """
        target = self._next_database()
        if target is None:
            return 0

        database, store = target
        pending = store.photos_without_caption(limit=SLICE_PHOTOS)
        if not pending:
            return 0

        settings = get_settings()
        captioner = Captioner.get(settings.caption_blip_model, num_threads=settings.caption_threads)
        texts = captioner.caption_images(
            [str(store.photo_path(photo)) for photo in pending], batch_size=len(pending)
        )

        written = store.set_caption_texts(
            {photo.photo_id: text for photo, text in zip(pending, texts) if text},
            model=settings.caption_blip_model,
        )
        if written:
            self._encode(store, pending, settings)
            # Покрытие в SQLite — то, что видит интерфейс. Без этого подписи
            # появляются, а индикатор прогресса стоит на месте.
            sync_stats(database["id"], store)

        self._last_run = time.monotonic()
        self.captioned_total += written
        covered, total = store.captions_coverage()
        print(f"[разметка] {database['name']}: {covered}/{total}")
        return written

    def _encode(self, store: IndexStore, photos, settings) -> None:
        """
        Кодирует только что появившиеся подписи. Сразу, а не в конце всей базы:
        иначе прерывание оставило бы тексты без векторов, покрытие в статистике
        было бы, а искать по нему всё равно нечем.
        """
        encoder = CaptionEncoder.get(settings.caption_model)
        fresh = [p for p in (store.get_photo(photo.photo_id) for photo in photos)
                 if p is not None and p.caption]
        if not fresh:
            return
        vectors = encoder.encode([p.caption for p in fresh])
        store.set_caption_vectors(
            {photo.photo_id: vector for photo, vector in zip(fresh, vectors)},
            model=settings.caption_model,
        )

    def _next_database(self) -> tuple[dict, IndexStore] | None:
        """Первая база, где есть неразмеченные снимки."""
        for database in db.list_captionable_databases():
            try:
                store = store_for(database)
            except (StoreError, OSError) as e:
                # база могла быть удалена между запросом к SQLite и открытием
                print(f"[разметка пропущена] {database['id']}: {e}")
                continue
            if store.photos_without_caption(limit=1):
                return database, store
        return None


caption_worker = CaptionWorker()
