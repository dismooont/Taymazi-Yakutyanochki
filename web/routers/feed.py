"""Персональная лента — см. web/feed.py про то, как она собирается."""

from __future__ import annotations

from fastapi import APIRouter

from web.deps import CurrentUser
from web.feed import build_feed
from web.schemas import SearchResultOut

router = APIRouter(prefix="/api/feed", tags=["feed"])


@router.get("", response_model=SearchResultOut)
def get_feed(user: CurrentUser) -> SearchResultOut:
    return SearchResultOut(results=build_feed(user))
