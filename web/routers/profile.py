"""
Профиль пользователя: лайкнутые и избранные фото со всех видимых баз (свои,
демо, чаты). Отдельный роутер, а не часть photos.py — здесь выдача не привязана
к одной базе.
"""

from __future__ import annotations

from fastapi import APIRouter

from web import db
from web.deps import CurrentUser
from web.routers.auth import _user_out
from web.schemas import ProfileOut, ProfilePhotoOut

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", response_model=ProfileOut)
def get_profile(user: CurrentUser) -> ProfileOut:
    return ProfileOut(
        user=_user_out(user),
        liked=[ProfilePhotoOut.from_row(row) for row in db.list_liked_photos(user["id"])],
        favorited=[ProfilePhotoOut.from_row(row) for row in db.list_favorite_photos(user["id"])],
    )
