"""
Очередь фоновых задач: импорт архива, пакетное добавление фото.

Почему поток, а не asyncio-таск. Работа здесь целиком блокирующая (CPU-инференс CLIP,
запись индекса на диск), а ставят задачи в очередь синхронные обработчики FastAPI,
которые сами выполняются в threadpool. Обычная queue.Queue плюс один рабочий поток
избавляют от возни с привязкой к event loop (call_soon_threadsafe и подобным) и точно
соответствуют требуемой модели «одна последовательная очередь на приложение»
(docs/WEB_PLAN.md, раздел 6).

Один воркер — сознательное ограничение: несколько параллельных индексаций всё равно
упёрлись бы в один и тот же лок модели, зато конкурировали бы за память.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

from web import db

PROGRESS_WRITE_INTERVAL = 0.5  # сек: чаще писать прогресс в SQLite бессмысленно


class JobCancelled(Exception):
    """Задача остановлена пользователем. Ловится воркером, не считается ошибкой."""


@dataclass
class JobContext:
    """То, что видит выполняемая функция: прогресс, отмена, сообщение для UI."""

    job_id: str
    user_id: str
    database_id: str | None
    _queue: "JobQueue"
    _last_write: float = 0.0

    def progress(self, done: int, total: int) -> None:
        self.check_cancelled()
        now = time.monotonic()
        # запись в БД дросселируется, но последний тик пишем всегда, иначе на экране
        # останется «998 из 1000» после завершения
        if now - self._last_write >= PROGRESS_WRITE_INTERVAL or done >= total:
            db.update_job_progress(self.job_id, done, total)
            self._last_write = now

    def set_message(self, message: str) -> None:
        db.set_job_message(self.job_id, message)

    def check_cancelled(self) -> None:
        if self._queue.is_cancelled(self.job_id):
            raise JobCancelled()


JobFunction = Callable[[JobContext], str | None]


class JobQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        # Новому воркеру — новая очередь. Иначе сентинел None, положенный прошлым stop()
        # и не забранный воркером (тот мог выйти раньше по флагу _stopping), достанется
        # новому воркеру первым же элементом и остановит его, не дав выполнить ни одной
        # задачи: очередь молча перестаёт работать, а задачи вечно висят в 'queued'.
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, args=(self._queue,), name="job-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._queue.put(None)  # разбудить воркер, если он ждёт
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------
    # Постановка и отмена
    # ------------------------------------------------------------------

    def submit(self, *, kind: str, user_id: str, database_id: str | None,
               function: JobFunction, total: int = 0) -> dict:
        job = db.create_job(user_id=user_id, database_id=database_id, kind=kind, total=total)
        self._queue.put((job["id"], function))
        return job

    def cancel(self, job_id: str) -> None:
        """
        Отмена кооперативная: флаг проверяется между батчами. Уже посчитанные фото
        остаются в базе — сказать «отменено, 300 из 1000 успели» честнее, чем откатывать
        десять минут работы.
        """
        with self._lock:
            self._cancelled.add(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.discard(job_id)

    # ------------------------------------------------------------------
    # Воркер
    # ------------------------------------------------------------------

    def _worker(self, own_queue: queue.Queue) -> None:
        # воркер держится за очередь, с которой был запущен: если приложение
        # перезапустят, он не станет разбирать очередь следующего экземпляра
        while not self._stopping.is_set():
            item = own_queue.get()
            if item is None:
                break
            job_id, function = item
            self._run_one(job_id, function)
            own_queue.task_done()

    def _run_one(self, job_id: str, function: JobFunction) -> None:
        job = db.get_job(job_id)
        if job is None:
            return

        if self.is_cancelled(job_id):
            db.finish_job(job_id, "error", "Отменено до начала выполнения")
            self._forget(job_id)
            return

        db.start_job(job_id)
        context = JobContext(
            job_id=job_id,
            user_id=job["user_id"],
            database_id=job["database_id"],
            _queue=self,
        )
        try:
            message = function(context)
            db.finish_job(job_id, "done", message or "Готово")
        except JobCancelled:
            db.finish_job(job_id, "error", "Отменено")
        except Exception as e:
            # трассировка — в лог сервера, пользователю только тип и текст:
            # внутренние пути и SQL в интерфейсе никому не нужны
            traceback.print_exc()
            db.finish_job(job_id, "error", f"{type(e).__name__}: {e}")
        finally:
            self._forget(job_id)


job_queue = JobQueue()


def recover_interrupted_jobs() -> int:
    """
    При старте помечает ошибкой задачи, оставшиеся в статусе running/queued от прошлого
    запуска. Дозапуска нет намеренно: пересобрать импорт заново проще и надёжнее, чем
    восстанавливать, на каком файле процесс упал.
    """
    return db.fail_unfinished_jobs("Прервано перезапуском сервера")
