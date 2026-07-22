/**
 * Отдельная страница снимка — не модалка: у неё свой адрес, значит на неё
 * можно дать ссылку, открыть в новой вкладке, вернуться кнопкой «назад»
 * браузера. Раньше открытие фото было оверлеем (PhotoLightbox); ушли от
 * этого по прямой просьбе — оверлей неудобно себя вёл при длинной ленте.
 */

import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, api, type Photo as PhotoInfo, type SearchHit } from '../api'
import { Empty, PhotoGrid, type Tile } from '../components'

function toTile(hit: SearchHit, databaseId: string): Tile {
  return {
    photoId: hit.photo_id,
    databaseId,
    thumbUrl: hit.thumb_url,
    fileUrl: hit.file_url,
    score: hit.score,
    caption: hit.caption,
    liked: hit.liked,
    favorited: hit.favorited,
    aiGenerated: hit.ai_generated,
  }
}

export function Photo() {
  const { id = '', photoId = '' } = useParams()
  const [photo, setPhoto] = useState<PhotoInfo | null>(null)
  const [similar, setSimilar] = useState<Tile[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setPhoto(null)
    setSimilar(null)
    setError(null)
    try {
      const [info] = await Promise.all([api.photo(id, photoId), api.viewPhoto(id, photoId)])
      setPhoto(info)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось открыть снимок')
      return
    }
    try {
      const result = await api.similar(id, photoId)
      setSimilar(result.results.map((hit) => toTile(hit, id)))
    } catch {
      setSimilar([])
    }
  }, [id, photoId])

  useEffect(() => {
    load()
  }, [load])

  const toggleLike = async () => {
    if (!photo) return
    const next = !photo.liked
    setPhoto({ ...photo, liked: next })
    try {
      await (next ? api.like(id, photoId) : api.unlike(id, photoId))
    } catch {
      setPhoto((current) => (current ? { ...current, liked: !next } : current))
    }
  }

  const toggleFavorite = async () => {
    if (!photo) return
    const next = !photo.favorited
    setPhoto({ ...photo, favorited: next })
    try {
      await (next ? api.favorite(id, photoId) : api.unfavorite(id, photoId))
    } catch {
      setPhoto((current) => (current ? { ...current, favorited: !next } : current))
    }
  }

  const applySimilarMark = (tile: Tile, patch: Partial<Tile>) => {
    setSimilar((current) =>
      current?.map((t) => (t.photoId === tile.photoId ? { ...t, ...patch } : t)) ?? current,
    )
  }

  const toggleSimilarLike = async (tile: Tile) => {
    const next = !tile.liked
    applySimilarMark(tile, { liked: next })
    try {
      await (next ? api.like(tile.databaseId, tile.photoId) : api.unlike(tile.databaseId, tile.photoId))
    } catch {
      applySimilarMark(tile, { liked: tile.liked })
    }
  }

  const toggleSimilarFavorite = async (tile: Tile) => {
    const next = !tile.favorited
    applySimilarMark(tile, { favorited: next })
    try {
      await (next
        ? api.favorite(tile.databaseId, tile.photoId)
        : api.unfavorite(tile.databaseId, tile.photoId))
    } catch {
      applySimilarMark(tile, { favorited: tile.favorited })
    }
  }

  if (error) {
    return (
      <div className="stack">
        <Link to={`/db/${id}`} className="eyebrow" style={{ textDecoration: 'none' }}>
          ← Назад к базе
        </Link>
        <Empty title={error} />
      </div>
    )
  }

  if (!photo) {
    return <div className="empty">Открываем снимок…</div>
  }

  return (
    <div className="stack" style={{ gap: 28 }}>
      <Link to={`/db/${id}`} className="eyebrow" style={{ textDecoration: 'none' }}>
        ← Назад к базе
      </Link>

      <div className="photo-page">
        <div className="photo-page__frame">
          <img className="photo-page__image" src={api.fileUrl(id, photoId)} alt="" />
          {photo.ai_generated && <span className="card__badge">Сгенерировано ИИ</span>}
        </div>

        <div className="photo-page__side stack">
          <div className="row">
            <button
              type="button"
              className="btn"
              data-active={photo.liked}
              onClick={toggleLike}
              aria-pressed={photo.liked}
            >
              {photo.liked ? '♥' : '♡'} {photo.liked ? 'Лайкнуто' : 'Лайк'}
            </button>
            <button
              type="button"
              className="btn"
              data-active={photo.favorited}
              onClick={toggleFavorite}
              aria-pressed={photo.favorited}
            >
              {photo.favorited ? '★' : '☆'} {photo.favorited ? 'В избранном' : 'В избранное'}
            </button>
          </div>
          {photo.caption && <p className="note">{photo.caption}</p>}
          <p className="mono" style={{ color: 'var(--muted)' }}>{photoId.slice(0, 8)}</p>
        </div>
      </div>

      <section className="stack">
        <p className="eyebrow">Похожие</p>
        {similar === null ? (
          <p className="note">Ищем похожие…</p>
        ) : similar.length === 0 ? (
          <Empty title="Похожих не нашлось" />
        ) : (
          <PhotoGrid tiles={similar} onToggleLike={toggleSimilarLike} onToggleFavorite={toggleSimilarFavorite} />
        )}
      </section>
    </div>
  )
}
