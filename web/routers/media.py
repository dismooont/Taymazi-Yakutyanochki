"""
Вкладка «Фильмы и музыка» — подборка по темам недавних запросов, лайков,
избранного и просмотров (see web.services.recent_theme_keywords).

Каждый каталог (OMDb, Last.fm) опрашивается независимо и может отсутствовать
(ключ не задан) или упасть по сети — ни то, ни другое не должно ронять ответ
целиком, просто у темы будет меньше разделов.
"""

from __future__ import annotations

from fastapi import APIRouter

from core import media as media_api
from web import services
from web.config import get_settings
from web.deps import CurrentUser
from web.schemas import ArtistOut, MediaOut, MediaThemeOut, MovieOut, TrackOut

router = APIRouter(prefix="/api/media", tags=["media"])

THEMES_LIMIT = 6


@router.get("", response_model=MediaOut)
def get_media(user: CurrentUser) -> MediaOut:
    settings = get_settings()
    if not settings.media_enabled:
        return MediaOut(enabled=False)

    keywords = services.recent_theme_keywords(user["id"], limit=THEMES_LIMIT)
    themes = [_theme_for(keyword, settings) for keyword in keywords]
    # Тема без единого результата ни в одном каталоге (редкое/непереводимое
    # слово) не несёт пользы на экране — только пустая карточка.
    themes = [t for t in themes if t.movies or t.tracks or t.artists]
    return MediaOut(enabled=True, themes=themes)


def _theme_for(keyword: str, settings) -> MediaThemeOut:
    movies: list[MovieOut] = []
    tracks: list[TrackOut] = []
    artists: list[ArtistOut] = []

    if settings.omdb_api_key:
        try:
            movies = [
                MovieOut(title=m.title, year=m.year, imdb_id=m.imdb_id, poster_url=m.poster_url)
                for m in media_api.search_movies(keyword, api_key=settings.omdb_api_key)
            ]
        except media_api.MediaApiError as e:
            print(f"[OMDb не ответил по теме {keyword!r}] {e}")

    if settings.lastfm_api_key:
        try:
            tracks = [
                TrackOut(name=t.name, artist=t.artist, url=t.url)
                for t in media_api.top_tracks_by_tag(keyword, api_key=settings.lastfm_api_key)
            ]
        except media_api.MediaApiError as e:
            print(f"[Last.fm треки не ответили по теме {keyword!r}] {e}")

        try:
            artists = [
                ArtistOut(name=a.name, url=a.url, image_url=a.image_url)
                for a in media_api.top_artists_by_tag(keyword, api_key=settings.lastfm_api_key)
            ]
        except media_api.MediaApiError as e:
            print(f"[Last.fm артисты не ответили по теме {keyword!r}] {e}")

    return MediaThemeOut(theme=keyword, movies=movies, tracks=tracks, artists=artists)
