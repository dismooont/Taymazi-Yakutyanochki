/**
 * Работа с выбранной базой: поиск, галерея, добавление и удаление снимков,
 * выгрузка архива. Объём базы виден в шапке постоянно — это требование сценария.
 */

import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, api, type Database, type SearchResult } from '../api'
import { Dropzone, Empty, JobProgress, PhotoGrid, Toast, type Tile, useJob } from '../components'
import { formatBytes, formatPhotos } from '../format'

type Tab = 'search' | 'gallery'
const PAGE_SIZE = 60

export function Workspace() {
  const { id = '' } = useParams()
  const [database, setDatabase] = useState<Database | null>(null)
  const [tab, setTab] = useState<Tab>('search')
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [gallery, setGallery] = useState<Tile[]>([])
  const [galleryTotal, setGalleryTotal] = useState(0)
  const [jobId, setJobId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  const refreshStats = useCallback(async () => {
    setDatabase(await api.stats(id))
  }, [id])

  const loadGallery = useCallback(
    async (offset = 0) => {
      const page = await api.photos(id, offset, PAGE_SIZE)
      setGalleryTotal(page.total)
      const tiles = page.items.map((photo) => ({
        photoId: photo.photo_id,
        thumbUrl: api.thumbUrl(id, photo.photo_id),
        fileUrl: api.fileUrl(id, photo.photo_id),
      }))
      setGallery((current) => (offset === 0 ? tiles : [...current, ...tiles]))
    },
    [id],
  )

  useEffect(() => {
    refreshStats().catch(() => setError('База не найдена'))
    // если пользователь ушёл со страницы во время индексации, прогресс должен
    // найтись сам после возвращения
    api
      .jobs(id)
      .then((jobs) => {
        const active = jobs.find((job) => job.status === 'queued' || job.status === 'running')
        if (active) setJobId(active.id)
      })
      .catch(() => undefined)
  }, [id, refreshStats])

  useEffect(() => {
    if (tab === 'gallery') loadGallery(0).catch(() => undefined)
  }, [tab, loadGallery])

  const job = useJob(jobId, () => {
    refreshStats().catch(() => undefined)
    if (tab === 'gallery') loadGallery(0).catch(() => undefined)
    setToast('Снимки добавлены')
  })

  const searchByText = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!query.trim()) return
    setBusy(true)
    setError(null)
    try {
      setResult(await api.searchText(id, query.trim()))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Поиск не удался')
    } finally {
      setBusy(false)
    }
  }

  const searchByImage = async (files: File[]) => {
    if (!files[0]) return
    setBusy(true)
    setError(null)
    try {
      setQuery('')
      setResult(await api.searchImage(id, files[0]))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Поиск не удался')
    } finally {
      setBusy(false)
    }
  }

  const addPhotos = async (files: File[]) => {
    setBusy(true)
    setError(null)
    try {
      const outcome = await api.addPhotos(id, files)
      if (outcome.job_id) {
        setJobId(outcome.job_id)
      } else {
        await refreshStats()
        if (tab === 'gallery') await loadGallery(0)
        const skipped = outcome.skipped.length
        setToast(
          `Добавлено снимков: ${outcome.added}` + (skipped ? `, пропущено: ${skipped}` : ''),
        )
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось добавить снимки')
    } finally {
      setBusy(false)
    }
  }

  const removePhoto = async (photoId: string) => {
    setGallery((current) => current.filter((tile) => tile.photoId !== photoId))
    setResult((current) =>
      current ? { ...current, results: current.results.filter((h) => h.photo_id !== photoId) } : current,
    )
    try {
      await api.deletePhoto(id, photoId)
      setGalleryTotal((total) => Math.max(0, total - 1))
      await refreshStats()
    } catch {
      setError('Снимок не удалён — обновите страницу')
    }
  }

  if (!database) {
    return <div className="empty">{error ?? 'Открываем базу…'}</div>
  }

  const searchTiles: Tile[] =
    result?.results.map((hit) => ({
      photoId: hit.photo_id,
      thumbUrl: hit.thumb_url,
      fileUrl: hit.file_url,
      score: hit.score,
    })) ?? []

  return (
    <div className="stack" style={{ gap: 28 }}>
      <header className="masthead">
        <div>
          <Link to="/" className="eyebrow" style={{ textDecoration: 'none' }}>
            ← Все базы
          </Link>
          <h1 className="title" style={{ marginTop: 6 }}>
            {database.name}
          </h1>
          <p className="mono" style={{ color: 'var(--muted)', margin: '6px 0 0' }}>
            {formatPhotos(database.photos_count)} · {formatBytes(database.total_bytes)}
          </p>
        </div>

        <div className="row">
          {!database.read_only && (
            <Dropzone accept="image/*" multiple variant="compact" onFiles={addPhotos}>
              Добавить снимки
            </Dropzone>
          )}
          <a className="btn" href={api.exportUrl(id)} download>
            Скачать архив
          </a>
        </div>
      </header>

      {job && <JobProgress job={job} onCancel={() => api.cancelJob(job.id).catch(() => undefined)} />}

      <div className="tabs" style={{ maxWidth: 320 }} role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'search'}
          className="tabs__item"
          onClick={() => setTab('search')}
        >
          Поиск
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'gallery'}
          className="tabs__item"
          onClick={() => setTab('gallery')}
        >
          Вся база
        </button>
      </div>

      {tab === 'search' && (
        <div className="stack">
          <form className="console" onSubmit={searchByText}>
            <div className="console__row">
              <input
                className="field"
                value={query}
                placeholder="рыжий кот на подоконнике"
                onChange={(event) => setQuery(event.target.value)}
                aria-label="Описание снимка"
              />
              <button type="submit" className="btn btn--primary" disabled={busy}>
                Найти
              </button>
            </div>

            <div className="console__alt">
              <span>Или найдите похожие по образцу:</span>
              <Dropzone accept="image/*" variant="compact" onFiles={searchByImage}>
                Перетащить снимок
              </Dropzone>
            </div>

            {result?.used_query && (
              <div className="console__readout">
                <span className="eyebrow">Ушло в модель</span>
                <span className="mono console__sent">{result.used_query}</span>
                <span className="topbar__spacer" />
                <span className="note">
                  Русский запрос переводится: CLIP обучен на английских подписях
                </span>
              </div>
            )}
          </form>

          {error && <p className="error">{error}</p>}

          {result && searchTiles.length === 0 && (
            <Empty
              title="Ничего не нашлось"
              hint={
                database.photos_count === 0
                  ? 'В базе пока нет снимков. Добавьте их — и поиск заработает.'
                  : 'Попробуйте описать снимок другими словами.'
              }
            />
          )}

          {searchTiles.length > 0 && (
            <PhotoGrid
              tiles={searchTiles}
              onRemove={database.read_only ? undefined : removePhoto}
            />
          )}

          {result && result.captions.length > 0 && (
            <div className="stack">
              <p className="eyebrow">Ближайшие подписи</p>
              {result.captions.map((caption, index) => (
                <p key={index} className="note">
                  <span className="mono" style={{ color: 'var(--signal)' }}>
                    {caption.score.toFixed(4)}
                  </span>{' '}
                  {caption.caption}
                </p>
              ))}
            </div>
          )}

          {!result && (
            <Empty
              title="Опишите, что ищете"
              hint="Например: «люди играют в теннис», «пицца на столе», «красный автобус»."
            />
          )}
        </div>
      )}

      {tab === 'gallery' && (
        <div className="stack">
          {gallery.length === 0 ? (
            <Empty title="В базе нет снимков" hint={database.read_only ? undefined : 'Добавьте их кнопкой в шапке.'} />
          ) : (
            <>
              <PhotoGrid tiles={gallery} onRemove={database.read_only ? undefined : removePhoto} />
              {gallery.length < galleryTotal && (
                <button
                  type="button"
                  className="btn"
                  onClick={() => loadGallery(gallery.length)}
                  style={{ alignSelf: 'center' }}
                >
                  Показать ещё ({galleryTotal - gallery.length})
                </button>
              )}
            </>
          )}
        </div>
      )}

      {toast && <Toast text={toast} onDone={() => setToast(null)} />}
    </div>
  )
}
