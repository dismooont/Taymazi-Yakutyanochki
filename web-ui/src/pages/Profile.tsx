/**
 * Профиль: лайкнутые и избранные снимки со всех видимых баз (свои, демо, чаты).
 * Лайк и избранное — независимые отметки (см. web/db.py), поэтому это два
 * отдельных списка, а не один с фильтром.
 */

import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api, type Profile as ProfileData, type ProfilePhoto } from '../api'
import { Empty, PhotoGrid, type Tile } from '../components'

const KIND_LABEL: Record<ProfilePhoto['database_kind'], string> = {
  personal: '',
  chat: 'база чата',
  demo: 'демо-база',
}

function source(photo: ProfilePhoto): string {
  const label = KIND_LABEL[photo.database_kind]
  return label ? `${photo.database_name} · ${label}` : photo.database_name
}

function toTile(photo: ProfilePhoto, mark: 'liked' | 'favorited'): Tile {
  return {
    photoId: photo.photo_id,
    databaseId: photo.database_id,
    thumbUrl: photo.thumb_url,
    fileUrl: photo.file_url,
    // тайл переиспользует слот подписи, чтобы показать базу-источник: у
    // профиля снимки собраны сразу из нескольких баз, и без этого непонятно,
    // где искать отмеченный снимок
    caption: source(photo),
    liked: mark === 'liked',
    favorited: mark === 'favorited',
  }
}

export function Profile() {
  const [data, setData] = useState<ProfileData | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await api.profile())
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось загрузить профиль')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  if (!data) {
    return <div className="empty">{error ?? 'Загружаем профиль…'}</div>
  }

  const unlike = async (tile: Tile) => {
    setData((current) =>
      current ? { ...current, liked: current.liked.filter((p) => p.photo_id !== tile.photoId || p.database_id !== tile.databaseId) } : current,
    )
    try {
      await api.unlike(tile.databaseId, tile.photoId)
    } catch {
      setError('Не удалось убрать лайк')
      refresh()
    }
  }

  const unfavorite = async (tile: Tile) => {
    setData((current) =>
      current
        ? {
            ...current,
            favorited: current.favorited.filter(
              (p) => p.photo_id !== tile.photoId || p.database_id !== tile.databaseId,
            ),
          }
        : current,
    )
    try {
      await api.unfavorite(tile.databaseId, tile.photoId)
    } catch {
      setError('Не удалось убрать из избранного')
      refresh()
    }
  }

  return (
    <div className="stack" style={{ gap: 28 }}>
      <header className="masthead">
        <div>
          <Link to="/databases" className="eyebrow" style={{ textDecoration: 'none' }}>
            ← Все базы
          </Link>
          <h1 className="title" style={{ marginTop: 6 }}>
            {data.user.display_name}
          </h1>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      <section className="stack">
        <p className="eyebrow">Лайки ({data.liked.length})</p>
        {data.liked.length === 0 ? (
          <Empty title="Пока нет лайков" hint="Отмечайте снимки сердечком в базе или в поиске." />
        ) : (
          <PhotoGrid
            tiles={data.liked.map((photo) => toTile(photo, 'liked'))}
            onToggleLike={unlike}
          />
        )}
      </section>

      <section className="stack">
        <p className="eyebrow">Избранное ({data.favorited.length})</p>
        {data.favorited.length === 0 ? (
          <Empty title="Пока нет избранного" hint="Отмечайте снимки звёздочкой в базе или в поиске." />
        ) : (
          <PhotoGrid
            tiles={data.favorited.map((photo) => toTile(photo, 'favorited'))}
            onToggleFavorite={unfavorite}
          />
        )}
      </section>
    </div>
  )
}
