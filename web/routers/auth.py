"""
Регистрация, вход, выход и смена пароля.

Правила, из-за которых код выглядит «избыточно осторожным», описаны в
docs/WEB_PLAN.md, раздел 7.1: одинаковый текст ошибки на неверный логин и неверный
пароль, постоянное время проверки, ограничение частоты попыток.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from web import db
from web.config import get_settings
from web.deps import CurrentUser, client_ip
from web.schemas import LoginRequest, PasswordChangeRequest, RegisterRequest, UserOut
from web.security import (
    SESSION_COOKIE,
    AuthError,
    hash_password,
    hash_token,
    login_limiter,
    needs_rehash,
    new_session_token,
    normalize_login,
    register_limiter,
    validate_login,
    validate_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

INVALID_CREDENTIALS = "Неверный логин или пароль"
# именованная константа starlette переехала между версиями — держим числом
HTTP_422 = 422


def _user_out(user: dict[str, Any]) -> UserOut:
    return UserOut(
        id=user["id"],
        login=user["login"],
        display_name=user["display_name"],
        has_password=bool(user["password_hash"]),
        has_telegram=_has_telegram(user["id"]),
    )


def _has_telegram(user_id: str) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM identities WHERE user_id = ? AND provider = 'telegram'", (user_id,)
        ).fetchone()
    return row is not None


def _start_session(response: Response, request: Request, user_id: str) -> None:
    settings = get_settings()
    token = new_session_token()
    db.create_session(
        user_id,
        hash_token(token),
        settings.session_ttl_days,
        request.headers.get("user-agent", ""),
    )
    db.touch_user(user_id)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, request: Request, response: Response) -> UserOut:
    settings = get_settings()
    if not settings.registration_open:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Регистрация закрыта")

    if not register_limiter.check(client_ip(request)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Слишком много регистраций, подождите")

    try:
        login = validate_login(payload.login)
        validate_password(payload.password, settings.min_password_length)
    except AuthError as e:
        raise HTTPException(HTTP_422, str(e)) from e

    if db.get_user_by_login(login) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Логин уже занят")

    user = db.create_user(
        login=login,
        display_name=(payload.display_name or payload.login).strip()[:64],
        password_hash=hash_password(payload.password),
        email=(payload.email or None),
    )
    _start_session(response, request, user["id"])
    return _user_out(user)


@router.post("/login", response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response) -> UserOut:
    login_name = normalize_login(payload.login)
    keys = (f"login:{login_name}", f"ip:{client_ip(request)}")
    if not all(login_limiter.check(key) for key in keys):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Слишком много попыток входа. Попробуйте через 15 минут",
        )

    user = db.get_user_by_login(login_name)
    # verify_password вызывается даже для несуществующего логина — постоянное время ответа
    if not verify_password(user["password_hash"] if user else None, payload.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)

    if needs_rehash(user["password_hash"]):
        db.set_password_hash(user["id"], hash_password(payload.password))

    for key in keys:
        login_limiter.reset(key)
    _start_session(response, request, user["id"])
    return _user_out(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, user: CurrentUser) -> Response:
    db.delete_session(request.state.session_token_hash)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.post("/logout-all")
def logout_all(request: Request, user: CurrentUser) -> dict:
    """Завершает все сессии, включая текущую: «выйти на всех устройствах»."""
    closed = db.delete_user_sessions(user["id"])
    return {"closed_sessions": closed}


me_router = APIRouter(prefix="/api/me", tags=["auth"])


@me_router.get("", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return _user_out(user)


@me_router.post("/password", response_model=UserOut)
def change_password(payload: PasswordChangeRequest, request: Request, user: CurrentUser) -> UserOut:
    settings = get_settings()

    # старый пароль не спрашиваем только у того, у кого его ещё нет
    # (аккаунт заведён через Telegram) — иначе задать пароль было бы невозможно
    if user["password_hash"] and not verify_password(user["password_hash"], payload.old_password):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Текущий пароль неверен")

    try:
        validate_password(payload.new_password, settings.min_password_length)
    except AuthError as e:
        raise HTTPException(HTTP_422, str(e)) from e

    db.set_password_hash(user["id"], hash_password(payload.new_password))
    # смена пароля разлогинивает остальные устройства: если пароль меняют из-за утечки,
    # украденная сессия должна умереть вместе с ним
    db.delete_user_sessions(user["id"], keep_token_hash=request.state.session_token_hash)
    return _user_out(db.get_user(user["id"]))
