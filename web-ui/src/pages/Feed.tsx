/**
 * Персональная лента — главная страница после входа (сценарий Pinterest).
 * Собирается на бэкенде (web/feed.py) из похожего на лайкнутое/просмотренное
 * и повтора недавних запросов; для нового пользователя без истории — из
 * недавно добавленных снимков видимых баз, чтобы экран не был пустым.
 */

import { useCallback, useEffect, useState } from 'react'
import { ApiError, api, type SearchHit } from '../api'
import { Empty, PhotoGrid, type Tile } from '../components'

function toTile(hit: SearchHit): Tile {
  return {
    photoId: hit.photo_id,
    databaseId: hit.database_id ?? '',
    thumbUrl: hit.thumb_url,
    fileUrl: hit.file_url,
    caption: hit.caption,
    liked: hit.liked,
    favorited: hit.favorited,
    aiGenerated: hit.ai_generated,
  }
}

export function Feed() {
  const [tiles, setTiles] = useState<Tile[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const result = await api.feed()
      setTiles(result.results.map(toTile))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось загрузить ленту')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const applyMark = (tile: Tile, patch: Partial<Tile>) => {
    setTiles((current) =>
      current?.map((t) =>
        t.photoId === tile.photoId && t.databaseId === tile.databaseId ? { ...t, ...patch } : t,
      ) ?? current,
    )
  }

  const toggleLike = async (tile: Tile) => {
    const next = !tile.liked
    applyMark(tile, { liked: next })
    try {
      await (next ? api.like(tile.databaseId, tile.photoId) : api.unlike(tile.databaseId, tile.photoId))
    } catch {
      applyMark(tile, { liked: tile.liked })
    }
  }

  const toggleFavorite = async (tile: Tile) => {
    const next = !tile.favorited
    applyMark(tile, { favorited: next })
    try {
      await (next
        ? api.favorite(tile.databaseId, tile.photoId)
        : api.unfavorite(tile.databaseId, tile.photoId))
    } catch {
      applyMark(tile, { favorited: tile.favorited })
    }
  }

  if (!tiles) {
    return <div className="empty">{error ?? 'Собираем ленту…'}</div>
  }

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="sakha-band" aria-hidden="true" />
      <header className="masthead">
        <div>
          <p className="eyebrow" style={{ marginTop: 0 }}>Лента</p>
          <h1 className="title" style={{ marginTop: 6 }}>Похоже, вам понравится</h1>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      {tiles.length === 0 ? (
        <Empty
          title="Пока нечего показать"
          hint="Полайкайте снимки или поищите что-нибудь в своих базах — лента подстроится."
        />
      ) : (
        <PhotoGrid tiles={tiles} onToggleLike={toggleLike} onToggleFavorite={toggleFavorite} />
      )}
    </div>
  )
}
