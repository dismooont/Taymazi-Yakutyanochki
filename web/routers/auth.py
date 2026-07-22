"""
Регистрация, вход, выход и смена пароля.

Правила, из-за которых код выглядит «избыточно осторожным», описаны в
docs/WEB_PLAN.md, раздел 7.1: одинаковый текст ошибки на неверный логин и неверный
пароль, постоянное время проверки, ограничение частоты попыток.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from web import db, telegram
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
    telegram_limiter,
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
        # Указывает на /api/me/avatar, а не на аватар конкретно этого id: ручка
        # всегда отдаёт аватар звонящего. _user_out сейчас и вызывается только
        # для «себя» (регистрация/вход/смена пароля) — если появится показ
        # чужих профилей, понадобится отдельная публичная ручка по user_id.
        avatar_url="/api/me/avatar" if avatar_path(user["id"]).exists() else None,
    )


def avatar_path(user_id: str) -> Path:
    return get_settings().user_dir(user_id) / "avatar.webp"


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


@router.post("/telegram", response_model=UserOut)
def telegram_auth(
    payload: dict, request: Request, response: Response,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> UserOut:
    """
    Вход или привязка через Telegram Login Widget.

    Три случая:
      1. Пользователь уже вошёл по паролю — привязываем Telegram к его аккаунту.
      2. Такой Telegram уже привязан — обычный вход.
      3. Ни того ни другого — заводим новый аккаунт без пароля; задать пароль
         можно позже через /api/me/password с пустым old_password.
    """
    settings = get_settings()
    if not settings.telegram_ready:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Вход через Telegram не настроен")

    if not telegram_limiter.check(client_ip(request)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Слишком много попыток, подождите")

    try:
        data = telegram.verify(payload, settings.telegram_bot_token)
    except telegram.TelegramAuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e

    linked = db.get_user_by_identity("telegram", data["telegram_id"])
    current = _user_from_cookie(session)

    if current is not None:
        if linked is not None and linked["id"] != current["id"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Этот Telegram уже привязан к другому аккаунту"
            )
        db.link_identity("telegram", data["telegram_id"], current["id"])
        return _user_out(db.get_user(current["id"]))

    if linked is None:
        linked = db.create_user(login=None, display_name=data["display_name"], password_hash=None)
        db.link_identity("telegram", data["telegram_id"], linked["id"])

    _start_session(response, request, linked["id"])
    return _user_out(linked)


def _user_from_cookie(session: str | None) -> dict[str, Any] | None:
    """Мягкая версия get_current_user: без сессии не ошибка, а просто «не вошёл»."""
    if not session:
        return None
    row = db.get_session(hash_token(session))
    if row is None or db.is_expired(row["expires_at"]):
        return None
    return db.get_user(row["user_id"])


me_router = APIRouter(prefix="/api/me", tags=["auth"])


@me_router.delete("/identities/telegram", response_model=UserOut)
def unlink_telegram(user: CurrentUser) -> UserOut:
    if not user["password_hash"]:
        # иначе аккаунт остался бы без единого способа войти
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Сначала задайте пароль — иначе войти будет нечем",
        )
    db.unlink_identity("telegram", user["id"])
    return _user_out(db.get_user(user["id"]))


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


# --------------------------------------------------------------------------
# Аватар
# --------------------------------------------------------------------------

AVATAR_SIZE = 256
MAX_AVATAR_BYTES = 5 * 1024 ** 2


@me_router.get("/avatar")
def get_own_avatar(user: CurrentUser) -> FileResponse:
    path = avatar_path(user["id"])
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Аватар не задан")
    # приватный кэш и покороче, чем у фото базы: аватар меняют, а имя файла
    # (avatar.webp) при этом не меняется, в отличие от снимков по хешу содержимого
    return FileResponse(path, headers={"Cache-Control": "private, max-age=300"})


@me_router.post("/avatar", response_model=UserOut)
def upload_avatar(user: CurrentUser, file: UploadFile = File(...)) -> UserOut:
    raw = file.file.read(MAX_AVATAR_BYTES + 1)
    if len(raw) > MAX_AVATAR_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Слишком большой файл")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Файл не является изображением"
        ) from e

    # квадратная обрезка по центру — в интерфейсе аватар всегда круглый,
    # некруглый исходник (панорама, портрет) иначе съезжал бы вбок
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image.thumbnail((AVATAR_SIZE, AVATAR_SIZE))

    path = avatar_path(user["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=85)
    return _user_out(db.get_user(user["id"]))


@me_router.delete("/avatar", response_model=UserOut)
def delete_avatar(user: CurrentUser) -> UserOut:
    avatar_path(user["id"]).unlink(missing_ok=True)
    return _user_out(db.get_user(user["id"]))
