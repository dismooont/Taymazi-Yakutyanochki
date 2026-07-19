"""
Настройки веб-приложения. Всё берётся из переменных окружения (.env в разработке,
env_file в docker-compose) — см. docs/WEB_PLAN.md, раздел 9.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def users_dir(self) -> Path:
        return self.data_dir / "users"

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
    )


def reset_settings() -> None:
    """Сбрасывает кэш настроек. Нужно тестам, которые подменяют переменные окружения."""
    get_settings.cache_clear()
