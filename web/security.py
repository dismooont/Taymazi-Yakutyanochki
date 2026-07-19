"""
Пароли, токены сессий и ограничение частоты попыток входа.

См. docs/WEB_PLAN.md, раздел 7.1 — здесь собрано всё, что относится к «мы теперь храним
чужие пароли, и это наша ответственность».
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

# Хеш заведомо недостижимого пароля. Нужен, чтобы проверка несуществующего логина
# занимала столько же времени, сколько проверка существующего: иначе по времени ответа
# перебираются зарегистрированные логины.
_DUMMY_HASH = _hasher.hash("dummy-password-for-constant-time-check")

LOGIN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")

# Пароли, которые ломаются первым же перебором. Список короткий намеренно: смысл
# не в полноте, а в том, чтобы отсечь самое очевидное.
COMMON_PASSWORDS = {
    "password", "password1", "password123", "qwerty123", "qwertyuiop", "1234567890",
    "123456789", "12345678910", "administrator", "iloveyou1", "letmein123", "welcome123",
    "parolparol", "adminadmin", "qweqweqwe", "1q2w3e4r5t", "zxcvbnmasd",
}


class AuthError(ValueError):
    """Ошибка валидации логина/пароля с готовым текстом для пользователя."""


# --------------------------------------------------------------------------
# Пароли
# --------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    """
    Проверяет пароль за постоянное время: если хеша нет (нет такого пользователя или
    у него вход только через Telegram), всё равно прогоняем argon2 по фиктивному хешу.
    """
    target = password_hash or _DUMMY_HASH
    try:
        _hasher.verify(target, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    return password_hash is not None


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return False


def normalize_login(login: str) -> str:
    return login.strip().lower()


def validate_login(login: str) -> str:
    """Возвращает нормализованный логин или бросает AuthError с понятным текстом."""
    normalized = normalize_login(login)
    if not LOGIN_RE.match(normalized):
        raise AuthError(
            "Логин: 3–32 символа, латиница, цифры и знаки _ . -, начинается с буквы или цифры"
        )
    return normalized


def validate_password(password: str, min_length: int) -> None:
    if len(password) < min_length:
        raise AuthError(f"Пароль должен быть не короче {min_length} символов")
    if password.lower() in COMMON_PASSWORDS:
        raise AuthError("Такой пароль слишком распространён — придумайте другой")


# --------------------------------------------------------------------------
# Сессии
# --------------------------------------------------------------------------

SESSION_COOKIE = "session"


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """
    В базе лежит sha256 токена, а не сам токен: утечка дампа БД не должна давать
    возможность войти под чужой сессией. Соль не нужна — токен и так случайный.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Ограничение частоты
# --------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """
    Скользящее окно в памяти. Приложение однопроцессное (docs/WEB_PLAN.md, раздел 2),
    поэтому внешнее хранилище не нужно; при перезапуске счётчики обнуляются — приемлемо.

    Блокировка временная и по паре (логин, IP) отдельно: если ограничивать только по IP,
    ботнет обходит лимит, а если только по логину — чужой аккаунт можно намеренно
    заблокировать, зная имя.
    """

    limit: int
    window_seconds: float
    _hits: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str) -> bool:
        """True, если попытка разрешена. Сама попытка при этом засчитывается."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > cutoff]
            allowed = len(hits) < self.limit
            if allowed:
                hits.append(now)
            self._hits[key] = hits
            return allowed

    def reset(self, key: str) -> None:
        """Успешный вход обнуляет счётчик, чтобы опечатки не копились до блокировки."""
        with self._lock:
            self._hits.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


login_limiter = RateLimiter(limit=5, window_seconds=15 * 60)
register_limiter = RateLimiter(limit=10, window_seconds=60 * 60)
