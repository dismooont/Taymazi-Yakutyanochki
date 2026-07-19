"""
Точка входа веб-приложения.

Запуск в разработке:
    uvicorn web.app:app --reload --port 8000

ВАЖНО: только один воркер. Веса CLIP занимают ~600 МБ на процесс, а несколько воркеров
писали бы в один и тот же FAISS-индекс наперегонки (docs/WEB_PLAN.md, раздел 2).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web import db
from web.config import get_settings
from web.jobs import job_queue, recover_interrupted_jobs
from web.routers import auth, databases, export, jobs, photos, search

# Мутирующие запросы обязаны нести этот заголовок. Простую HTML-форму с чужого сайта
# так не подделать: заголовок требует XHR/fetch, а на них распространяется CORS.
# Вместе с SameSite=Lax это закрывает CSRF без токенов в каждой форме.
CSRF_HEADER = "x-requested-with"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT_PATHS = {"/api/health"}


def _preload_model() -> None:
    """
    Прогревает CLIP на старте.

    Без этого веса грузятся при первом же запросе, которому нужна размерность вектора, —
    например при создании пустой базы, — и пользователь ждёт 15+ секунд на действии,
    которое должно быть мгновенным. Ошибку загрузки не считаем фатальной: список баз
    и вход работают и без модели, а причина будет видна в логе.
    """
    from core.model import ModelHolder

    try:
        holder = ModelHolder.get()
        print(f"Модель готова: dim={holder.dim}, device={holder.device}")
    except Exception as e:
        print(f"[ВНИМАНИЕ] модель не загружена: {e}\n"
              f"Поиск и добавление фото работать не будут, пока это не исправлено.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.users_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    removed = db.purge_expired_sessions()
    if removed:
        print(f"Удалено протухших сессий: {removed}")
    _preload_model()

    interrupted = recover_interrupted_jobs()
    if interrupted:
        print(f"Задач прервано прошлым перезапуском: {interrupted}")
    job_queue.start()
    print(f"Данные: {settings.data_dir} | регистрация: "
          f"{'открыта' if settings.registration_open else 'закрыта'} | "
          f"Telegram-вход: {'вкл' if settings.telegram_auth_enabled else 'выкл'}")
    try:
        yield
    finally:
        job_queue.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="CLIP Search API", version="0.1.0", lifespan=lifespan)

    # credentials=True требует явного списка источников — со звёздочкой браузер
    # откажется слать cookie
    origins = {settings.public_url, "http://localhost:5173", "http://127.0.0.1:5173"}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_csrf_header(request: Request, call_next):
        if request.method not in SAFE_METHODS and request.url.path not in CSRF_EXEMPT_PATHS:
            if CSRF_HEADER not in request.headers:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Отсутствует заголовок X-Requested-With"},
                )
        return await call_next(request)

    app.include_router(auth.router)
    app.include_router(auth.me_router)
    app.include_router(databases.router)
    app.include_router(databases.quota_router)
    app.include_router(photos.router)
    app.include_router(search.router)
    app.include_router(export.router)
    app.include_router(jobs.router)

    @app.get("/api/health", tags=["service"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config", tags=["service"])
    def public_config() -> dict:
        """
        Настройки, нужные интерфейсу до входа: показывать ли кнопку Telegram и
        открыта ли регистрация. Ничего секретного здесь быть не должно — ручка
        доступна без авторизации.
        """
        settings = get_settings()
        return {
            "registration_open": settings.registration_open,
            "telegram_auth": settings.telegram_ready,
            "telegram_bot": settings.telegram_bot_username if settings.telegram_ready else None,
        }

    return app


app = create_app()
