"""
Настройки веб-приложения. Всё берётся из переменных окружения (.env в разработке,
env_file в docker-compose) — см. docs/WEB_PLAN.md, раздел 9.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.captions import DEFAULT_CAPTION_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# В Docker переменные приходят через env_file, а при локальном запуске их неоткуда
# взять: без этого всё, что описано в .env.example, молча не применялось бы —
# uvicorn просто не знает про .env.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    public_url: str
    registration_open: bool
    telegram_auth_enabled: bool
    session_ttl_days: int
    max_db_per_user: int
    max_photos_per_db: int
    max_bytes_per_user: int
    min_password_length: int
    trust_proxy: bool
    telegram_bot_token: str
    telegram_bot_username: str
    telegram_proxy: str
    service_token: str
    caption_search_enabled: bool
    caption_model: str
    caption_auto_enabled: bool
    caption_idle_seconds: float
    caption_force_after: float
    caption_threads: int | None
    caption_blip_model: str

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def users_dir(self) -> Path:
        return self.data_dir / "users"

    @property
    def demo_index_dir(self) -> Path:
        """
        Индекс COCO, построенный CLI (команда build). Подключается как демо-база
        только для чтения: у пользовательских баз нет подписей, поэтому поиск
        «фото → подпись» больше нигде не показать.
        """
        return Path(os.environ.get("DEMO_INDEX_DIR", PROJECT_ROOT / "index")).resolve()

    @property
    def cookie_secure(self) -> bool:
        """
        Флаг Secure у cookie выводится из адреса, а не задаётся отдельной переменной:
        так его нельзя случайно забыть включить в проде и невозможно сломать разработку
        на http://localhost, где браузер такую cookie просто не сохранит.
        """
        return self.public_url.startswith("https://")

    @property
    def telegram_ready(self) -> bool:
        """
        Вход через Telegram работает, только если включён И есть чем проверять подпись
        И известно имя бота для виджета. Полувключённое состояние хуже выключенного:
        кнопка есть, а вход не работает.
        """
        return bool(
            self.telegram_auth_enabled and self.telegram_bot_token and self.telegram_bot_username
        )

    def user_dir(self, user_id: str) -> Path:
        return self.users_dir / user_id

    def database_dir(self, user_id: str, database_id: str) -> Path:
        return self.user_dir(user_id) / "databases" / database_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
    return Settings(
        data_dir=data_dir,
        public_url=os.environ.get("PUBLIC_URL", "http://localhost:5173"),
        registration_open=_flag("REGISTRATION_OPEN", True),
        telegram_auth_enabled=_flag("TELEGRAM_AUTH_ENABLED", False),
        session_ttl_days=_int("SESSION_TTL_DAYS", 30),
        max_db_per_user=_int("MAX_DB_PER_USER", 5),
        max_photos_per_db=_int("MAX_PHOTOS_PER_DB", 5000),
        max_bytes_per_user=_int("MAX_BYTES_PER_USER", 3 * 1024 ** 3),
        min_password_length=_int("MIN_PASSWORD_LENGTH", 10),
        # Включать ТОЛЬКО когда перед приложением действительно стоит обратный прокси,
        # который сам проставляет X-Real-IP. Иначе клиент подделает заголовок и обойдёт
        # ограничение частоты входа.
        trust_proxy=_flag("TRUST_PROXY", False),
        # Токен того же бота, что и в Telegram-боте: подпись виджета проверяется им.
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        # Имя бота нужно фронтенду, чтобы отрисовать виджет входа.
        telegram_bot_username=os.environ.get("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@"),
        telegram_proxy=os.environ.get("TELEGRAM_PROXY", "").strip(),
        # Токен, которым Telegram-бот доказывает API, что он свой. Пока не задан,
        # ручки /api/bot не существует вовсе — так безопаснее умолчания.
        service_token=os.environ.get("SERVICE_TOKEN", "").strip(),
        # Поиск по подписям выключен по умолчанию: он тянет вторую модель (~90 МБ
        # весов) и имеет смысл только тогда, когда подписи уже сгенерированы.
        # Замер пользы — docs/CAPTION_SEARCH.md.
        caption_search_enabled=_flag("CAPTION_SEARCH_ENABLED", False),
        caption_model=os.environ.get("CAPTION_MODEL", DEFAULT_CAPTION_MODEL).strip(),
        # Фоновая генерация подписей (C4). Отдельный флаг от caption_search: искать
        # можно и по подписям, сгенерированным заранее командой, не запуская BLIP
        # в вебе вовсе.
        caption_auto_enabled=_flag("CAPTION_AUTO_ENABLED", False),
        # Порог простоя: столько секунд без запросов, чтобы начать размечать.
        caption_idle_seconds=float(_int("CAPTION_IDLE_SECONDS", 30)),
        # Раз в столько секунд размечаем даже под нагрузкой — иначе активный
        # пользователь не даст разметить свою базу никогда.
        caption_force_after=float(_int("CAPTION_FORCE_AFTER", 600)),
        # Сколько ядер отдать BLIP. None — не ограничивать. Меньшее число замедляет
        # разметку, зато оставляет поиску воздух.
        caption_threads=(_int("CAPTION_THREADS", 0) or None),
        caption_blip_model=os.environ.get(
            "CAPTION_BLIP_MODEL", "Salesforce/blip-image-captioning-base"
        ).strip(),
    )


def reset_settings() -> None:
    """Сбрасывает кэш настроек. Нужно тестам, которые подменяют переменные окружения."""
    get_settings.cache_clear()
