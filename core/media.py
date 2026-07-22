"""
Клиенты внешних каталогов для вкладки «Фильмы и музыка» (web/routers/media.py):
OMDb — поиск фильмов по названию/теме, Last.fm — топ треков и артистов по тегу.

Оба — простые синхронные GET с JSON-ответом, без SDK: экономит зависимость ради
двух вызываемых на весь модуль методов. Сетевой сбой одного каталога не должен
ронять другой — поэтому здесь только транспорт, решение "пропустить и продолжить"
принимает вызывающий код (web/routers/media.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

OMDB_URL = "http://www.omdbapi.com/"
LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"

REQUEST_TIMEOUT = 10.0


class MediaApiError(RuntimeError):
    """Каталог отклонил запрос или не ответил вовремя."""


@dataclass(frozen=True)
class MovieResult:
    title: str
    year: str
    imdb_id: str
    poster_url: str | None


@dataclass(frozen=True)
class TrackResult:
    name: str
    artist: str
    url: str


@dataclass(frozen=True)
class ArtistResult:
    name: str
    url: str
    image_url: str | None


def search_movies(query: str, *, api_key: str, limit: int = 6) -> list[MovieResult]:
    try:
        response = requests.get(
            OMDB_URL,
            params={"apikey": api_key, "s": query, "type": "movie"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise MediaApiError(f"OMDb не ответил: {e}") from e

    if not response.ok:
        raise MediaApiError(f"OMDb отклонил запрос ({response.status_code})")

    data = response.json()
    if data.get("Response") != "True":
        return []  # "Movie not found!" и т.п. — не ошибка, просто пусто по этой теме

    results = []
    for item in data.get("Search", [])[:limit]:
        poster = item.get("Poster")
        results.append(
            MovieResult(
                title=item.get("Title", ""),
                year=item.get("Year", ""),
                imdb_id=item.get("imdbID", ""),
                poster_url=poster if poster and poster != "N/A" else None,
            )
        )
    return results


def _lastfm_get(method: str, *, api_key: str, tag: str, limit: int) -> dict:
    try:
        response = requests.get(
            LASTFM_URL,
            params={
                "method": method,
                "tag": tag,
                "api_key": api_key,
                "format": "json",
                "limit": limit,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise MediaApiError(f"Last.fm не ответил: {e}") from e

    if not response.ok:
        raise MediaApiError(f"Last.fm отклонил запрос ({response.status_code})")

    data = response.json()
    if "error" in data:
        raise MediaApiError(f"Last.fm сообщил об ошибке: {data.get('message', data['error'])}")
    return data


# У артиста без реального фото Last.fm вместо пустой строки отдаёт свою
# картинку-заглушку («звезда» на сером фоне) — с виду валидный URL, ничем не
# помеченный как отсутствие фото. Хеш файла один и тот же для всех отсутствующих
# картинок и известен по многочисленным интеграциям с этим API.
_LASTFM_PLACEHOLDER_HASH = "2a96cbd8b46e442fc41c2b86b821562f"


def _lastfm_image(images: list[dict]) -> str | None:
    """Last.fm отдаёт несколько размеров подряд — берём последний (обычно самый крупный)."""
    for image in reversed(images or []):
        url = image.get("#text")
        if url and _LASTFM_PLACEHOLDER_HASH not in url:
            return url
    return None


def top_tracks_by_tag(tag: str, *, api_key: str, limit: int = 6) -> list[TrackResult]:
    data = _lastfm_get("tag.gettoptracks", api_key=api_key, tag=tag, limit=limit)
    tracks = data.get("tracks", {}).get("track", [])
    return [
        TrackResult(
            name=track.get("name", ""),
            artist=(track.get("artist") or {}).get("name", ""),
            url=track.get("url", ""),
        )
        for track in tracks
    ]


def top_artists_by_tag(tag: str, *, api_key: str, limit: int = 6) -> list[ArtistResult]:
    data = _lastfm_get("tag.gettopartists", api_key=api_key, tag=tag, limit=limit)
    artists = data.get("topartists", {}).get("artist", [])
    return [
        ArtistResult(
            name=artist.get("name", ""),
            url=artist.get("url", ""),
            image_url=_lastfm_image(artist.get("image", [])),
        )
        for artist in artists
    ]
