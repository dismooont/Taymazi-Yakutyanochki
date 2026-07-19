/** Общие элементы интерфейса. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Job } from './api'
import { formatScore } from './format'

/* ------------------------------------------------------------------ */
/* Прогресс фоновой задачи                                             */
/* ------------------------------------------------------------------ */

/**
 * Следит за задачей опросом раз в секунду. Опрос, а не SSE: на пользователя
 * приходится одна активная задача, и поллинг переживает разрыв соединения
 * без дополнительного кода.
 */
export function useJob(jobId: string | null, onFinish?: () => void) {
  const [job, setJob] = useState<Job | null>(null)
  const finished = useRef(false)

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      return
    }
    finished.current = false
    let alive = true

    const tick = async () => {
      try {
        const next = await api.job(jobId)
        if (!alive) return
        setJob(next)
        if ((next.status === 'done' || next.status === 'error') && !finished.current) {
          finished.current = true
          onFinish?.()
        }
      } catch {
        /* сеть моргнула — попробуем на следующем тике */
      }
    }

    tick()
    const timer = setInterval(tick, 1000)
    return () => {
      alive = false
      clearInterval(timer)
    }
    // onFinish намеренно не в зависимостях: пересоздание колбэка не должно
    // перезапускать опрос
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  return job
}

export function JobProgress({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const done = job.status === 'done' || job.status === 'error'
  const percent = job.progress_total > 0 ? (job.progress_done / job.progress_total) * 100 : 0

  const headline =
    job.status === 'queued'
      ? job.queue_position > 0
        ? `В очереди, перед вами задач: ${job.queue_position}`
        : 'В очереди'
      : job.status === 'running'
        ? job.message || 'Обработка'
        : job.message || (job.status === 'done' ? 'Готово' : 'Не удалось')

  return (
    <div className="job" role="status" aria-live="polite">
      <div className="row">
        <strong>{headline}</strong>
        <span className="topbar__spacer" />
        {job.progress_total > 0 && !done && (
          <span className="mono">
            {job.progress_done} / {job.progress_total}
          </span>
        )}
        {!done && onCancel && (
          <button type="button" className="btn btn--quiet" onClick={onCancel}>
            Остановить
          </button>
        )}
      </div>
      {!done && (
        <div className="job__bar">
          <div className="job__fill" style={{ width: `${percent}%` }} />
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Зона перетаскивания                                                 */
/* ------------------------------------------------------------------ */

/**
 * Зона перетаскивания в двух видах: panel — крупная карточка для экрана выбора базы,
 * compact — обычная кнопка, которая вдобавок принимает перетаскивание. Без compact
 * кнопка «Добавить снимки» раздувалась в карточку и разносила шапку.
 */
export function Dropzone({
  accept,
  multiple,
  onFiles,
  children,
  variant = 'panel',
}: {
  accept: string
  multiple?: boolean
  onFiles: (files: File[]) => void
  children: React.ReactNode
  variant?: 'panel' | 'compact'
}) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const take = useCallback(
    (list: FileList | null) => {
      const files = Array.from(list ?? [])
      if (files.length) onFiles(multiple ? files : files.slice(0, 1))
    },
    [multiple, onFiles],
  )

  return (
    <>
      <button
        type="button"
        className={variant === 'panel' ? 'starter' : 'btn drop'}
        data-over={over}
        onClick={() => input.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault()
          setOver(false)
          take(event.dataTransfer.files)
        }}
      >
        {children}
      </button>
      <input
        ref={input}
        type="file"
        accept={accept}
        multiple={multiple}
        hidden
        onChange={(event) => {
          take(event.target.files)
          event.target.value = '' // тот же файл можно выбрать повторно
        }}
      />
    </>
  )
}

/* ------------------------------------------------------------------ */
/* Сетка снимков                                                       */
/* ------------------------------------------------------------------ */

export interface Tile {
  photoId: string
  thumbUrl: string
  fileUrl: string
  score?: number
}

export function PhotoGrid({
  tiles,
  onRemove,
}: {
  tiles: Tile[]
  onRemove?: (photoId: string) => void
}) {
  const [zoomed, setZoomed] = useState<string | null>(null)
  const best = tiles.reduce((max, tile) => Math.max(max, tile.score ?? 0), 0)

  useEffect(() => {
    if (!zoomed) return
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && setZoomed(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomed])

  return (
    <>
      <div className="grid">
        {tiles.map((tile) => (
          <figure className="card" key={tile.photoId} style={{ margin: 0 }}>
            <button
              type="button"
              className="card__frame"
              onClick={() => setZoomed(tile.fileUrl)}
              aria-label="Открыть снимок целиком"
            >
              <img src={tile.thumbUrl} alt="" loading="lazy" />
            </button>

            {tile.score !== undefined && (
              <div className="proximity">
                {/* длина линии — близость относительно первого места в этой выдаче */}
                <div
                  className="proximity__fill"
                  style={{ width: `${best > 0 ? (tile.score / best) * 100 : 0}%` }}
                />
              </div>
            )}

            <figcaption className="card__foot">
              <span className="card__id">{tile.photoId.slice(0, 8)}</span>
              {tile.score !== undefined && (
                <span className="card__score" title="Косинусная близость к запросу">
                  {formatScore(tile.score)}
                </span>
              )}
            </figcaption>

            {onRemove && (
              <button
                type="button"
                className="card__remove"
                onClick={() => onRemove(tile.photoId)}
                aria-label="Удалить снимок из базы"
              >
                ×
              </button>
            )}
          </figure>
        ))}
      </div>

      {zoomed && (
        <button type="button" className="lightbox" onClick={() => setZoomed(null)} aria-label="Закрыть">
          <img src={zoomed} alt="" />
        </button>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/* Мелочи                                                              */
/* ------------------------------------------------------------------ */

export function Toast({ text, onDone }: { text: string; onDone: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onDone, 5000)
    return () => clearTimeout(timer)
  }, [text, onDone])

  return (
    <div className="toast" role="status">
      {text}
    </div>
  )
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <p className="eyebrow" style={{ marginTop: 0 }}>
        {title}
      </p>
      {hint && <p className="note">{hint}</p>}
    </div>
  )
}
