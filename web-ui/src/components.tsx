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
  onEmptyDrop,
  children,
  variant = 'panel',
}: {
  accept: string
  multiple?: boolean
  onFiles: (files: File[]) => void
  /**
   * Перетащили что-то, в чём нет файла — обычно так бывает, когда тащат
   * <img> прямо со страницы (из ленты, галереи): браузер в этом случае не
   * кладёт файл в dataTransfer вовсе, это ограничение самого браузера, а не
   * баг. Без колбэка такой drop молча ничего не делает — выглядит как
   * «кнопка не работает».
   */
  onEmptyDrop?: () => void
  children: React.ReactNode
  variant?: 'panel' | 'compact'
}) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const take = useCallback(
    (list: FileList | null) => {
      const files = Array.from(list ?? [])
      if (files.length) onFiles(multiple ? files : files.slice(0, 1))
      else onEmptyDrop?.()
    },
    [multiple, onFiles, onEmptyDrop],
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
/* Область для перетаскивания и вставки из буфера                      */
/* ------------------------------------------------------------------ */

/**
 * Крупная зона приёма снимков: перетаскивание, выбор файлом и вставка из буфера.
 *
 * Вставка слушается на всём документе, а не только на самой зоне: чтобы поймать
 * событие на элементе, его пришлось бы сначала сфокусировать, а человек, нажимая
 * Ctrl+V, ни на что не нажимал — он только что сделал снимок экрана.
 *
 * Из буфера берутся только элементы-файлы с типом image/*. Обычный текст сюда
 * не попадает, поэтому вставка в поле поиска продолжает работать как обычно.
 */
export function DropArea({
  onFiles,
  disabled,
  hint,
}: {
  onFiles: (files: File[]) => void
  disabled?: boolean
  hint?: string
}) {
  const [over, setOver] = useState(false)
  const [flash, setFlash] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (disabled) return

    const onPaste = (event: ClipboardEvent) => {
      const files = Array.from(event.clipboardData?.items ?? [])
        .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
        .map((item) => item.getAsFile())
        .filter((file): file is File => file !== null)
      if (!files.length) return

      event.preventDefault()
      // у снимка экрана имени нет — подставляем своё, иначе на сервер уйдёт «blob»
      const named = files.map((file, index) =>
        file.name && file.name !== 'image.png'
          ? file
          : new File([file], `вставка-${Date.now()}-${index + 1}.png`, { type: file.type }),
      )
      setFlash(`Вставлено из буфера: ${named.length}`)
      onFiles(named)
    }

    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  }, [onFiles, disabled])

  useEffect(() => {
    if (!flash) return
    const timer = setTimeout(() => setFlash(null), 2500)
    return () => clearTimeout(timer)
  }, [flash])

  if (disabled) return null

  const take = (list: FileList | null) => {
    const files = Array.from(list ?? []).filter((file) => file.type.startsWith('image/'))
    if (files.length) onFiles(files)
  }

  return (
    <>
      <button
        type="button"
        className="droparea"
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
        <span className="droparea__title">
          {over ? 'Отпустите — заберу' : 'Перетащите снимки сюда'}
        </span>
        <span className="note">
          {flash ?? hint ?? 'Или нажмите, чтобы выбрать файлы. Скриншот можно вставить: Ctrl+V'}
        </span>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
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
  /** нужен для лайка/избранного: они привязаны к конкретной базе, а не только к снимку */
  databaseId: string
  thumbUrl: string
  fileUrl: string
  score?: number
  caption?: string
  liked?: boolean
  favorited?: boolean
  /** снимок сгенерирован YandexART, потому что поиск ничего не нашёл */
  aiGenerated?: boolean
}

/** Подпись под снимком: показ и, если разрешено, редактирование на месте. */
function CaptionCell({
  caption,
  onSave,
}: {
  caption: string
  onSave?: (caption: string) => Promise<void> | void
}) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(caption)
  const [busy, setBusy] = useState(false)

  // если подпись сменилась извне (перезагрузка галереи), подхватываем её
  useEffect(() => setValue(caption), [caption])

  if (!onSave) {
    return caption ? (
      <p className="card__caption" title={caption}>
        {caption}
      </p>
    ) : null
  }

  if (!editing) {
    return caption ? (
      <button
        type="button"
        className="card__caption card__caption--edit"
        title="Изменить подпись"
        onClick={() => setEditing(true)}
      >
        {caption}
      </button>
    ) : (
      <button type="button" className="card__caption-add" onClick={() => setEditing(true)}>
        + подпись
      </button>
    )
  }

  const commit = async () => {
    if (value.trim() === caption) {
      setEditing(false)
      return
    }
    setBusy(true)
    try {
      await onSave(value.trim())
      setEditing(false)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card__caption-edit">
      <textarea
        className="field card__caption-input"
        value={value}
        autoFocus
        rows={2}
        maxLength={500}
        placeholder="Опишите снимок своими словами"
        disabled={busy}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            commit()
          }
          if (event.key === 'Escape') {
            setValue(caption)
            setEditing(false)
          }
        }}
      />
      <div className="row" style={{ gap: 6 }}>
        <button type="button" className="btn btn--primary" disabled={busy} onClick={commit}>
          Сохранить
        </button>
        <button
          type="button"
          className="btn"
          disabled={busy}
          onClick={() => {
            setValue(caption)
            setEditing(false)
          }}
        >
          Отмена
        </button>
      </div>
    </div>
  )
}

export function PhotoGrid({
  tiles,
  onRemove,
  onEditCaption,
  onToggleLike,
  onToggleFavorite,
  fused = false,
}: {
  tiles: Tile[]
  onRemove?: (photoId: string) => void
  /** Задать/изменить подпись снимка. Если не передан — подписи только для чтения. */
  onEditCaption?: (photoId: string, caption: string) => Promise<void> | void
  /** Лайк и избранное — независимые отметки. Не переданы — кнопки не показываются. */
  onToggleLike?: (tile: Tile) => void
  onToggleFavorite?: (tile: Tile) => void
  /** Выдача получена слиянием с поиском по подписям — оценка тогда не косинус. */
  fused?: boolean
}) {
  const [zoomed, setZoomed] = useState<Tile | null>(null)

  // Обычная оценка — косинус, он неотрицателен, и полоску можно мерить от нуля.
  // Оценка слияния — взвешенная сумма отклонений от среднего, и она свободно
  // уходит в минус: у половины выдачи она отрицательна по построению. Меряя её
  // от нуля, мы получили бы отрицательную ширину, то есть пустую полоску у всего
  // нижнего хвоста.
  const scores = tiles.map((tile) => tile.score ?? 0)
  const best = scores.length ? Math.max(...scores) : 0
  const floor = fused && scores.length ? Math.min(...scores) : 0
  const span = best - floor
  const fill = (score: number) => (span > 0 ? ((score - floor) / span) * 100 : 0)

  useEffect(() => {
    if (!zoomed) return
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && setZoomed(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomed])

  // Открытый снимок мог с тех пор поменяться в исходном списке (лайк/избранное
  // из этой же сетки) — берём свежую версию оттуда, если она там ещё есть.
  const zoomedLive = zoomed
    ? tiles.find((t) => t.photoId === zoomed.photoId && t.databaseId === zoomed.databaseId) ?? zoomed
    : null

  return (
    <>
      <div className="grid">
        {tiles.map((tile) => (
          <figure className="card" key={tile.photoId} style={{ margin: 0 }}>
            <button
              type="button"
              className="card__frame"
              onClick={() => setZoomed(tile)}
              aria-label="Открыть снимок целиком"
            >
              <img src={tile.thumbUrl} alt="" loading="lazy" />
              {tile.aiGenerated && <span className="card__badge">Сгенерировано ИИ</span>}
            </button>

            {(onToggleLike || onToggleFavorite) && (
              <div className="card__marks">
                {onToggleLike && (
                  <button
                    type="button"
                    className="card__mark"
                    data-active={Boolean(tile.liked)}
                    aria-pressed={Boolean(tile.liked)}
                    aria-label={tile.liked ? 'Убрать лайк' : 'Поставить лайк'}
                    title={tile.liked ? 'Убрать лайк' : 'Лайк'}
                    onClick={() => onToggleLike(tile)}
                  >
                    {tile.liked ? '♥' : '♡'}
                  </button>
                )}
                {onToggleFavorite && (
                  <button
                    type="button"
                    className="card__mark"
                    data-active={Boolean(tile.favorited)}
                    aria-pressed={Boolean(tile.favorited)}
                    aria-label={tile.favorited ? 'Убрать из избранного' : 'В избранное'}
                    title={tile.favorited ? 'Убрать из избранного' : 'В избранное'}
                    onClick={() => onToggleFavorite(tile)}
                  >
                    {tile.favorited ? '★' : '☆'}
                  </button>
                )}
              </div>
            )}

            {tile.score !== undefined && (
              <div className="proximity">
                {/* длина линии — близость относительно первого места в этой выдаче */}
                <div className="proximity__fill" style={{ width: `${fill(tile.score)}%` }} />
              </div>
            )}

            <figcaption className="card__foot">
              <span className="card__id">{tile.photoId.slice(0, 8)}</span>
              {tile.score !== undefined && (
                <span
                  className="card__score"
                  title={
                    fused
                      ? 'Оценка слияния: поиск по снимку и по подписи вместе'
                      : 'Косинусная близость к запросу'
                  }
                >
                  {formatScore(tile.score)}
                </span>
              )}
            </figcaption>

            <CaptionCell
              caption={tile.caption ?? ''}
              onSave={onEditCaption ? (caption) => onEditCaption(tile.photoId, caption) : undefined}
            />

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

      {zoomedLive && (
        <PhotoLightbox
          tile={zoomedLive}
          onClose={() => setZoomed(null)}
          onOpenSimilar={setZoomed}
          onToggleLike={onToggleLike && ((t) => {
            onToggleLike(t)
            // отмечаем сразу и локально: карточка в сетке обновится, только
            // когда родитель перерисует tiles, а до этого клик по сердечку
            // в лайтбоксе видел бы старое значение и слал бы одно и то же
            // действие раз за разом (баг, который это чинит)
            setZoomed((z) => (z && z.photoId === t.photoId && z.databaseId === t.databaseId
              ? { ...z, liked: !z.liked } : z))
          })}
          onToggleFavorite={onToggleFavorite && ((t) => {
            onToggleFavorite(t)
            setZoomed((z) => (z && z.photoId === t.photoId && z.databaseId === t.databaseId
              ? { ...z, favorited: !z.favorited } : z))
          })}
        />
      )}
    </>
  )
}

/**
 * Открытая целиком фотография вместе с похожими — «зашёл в снимок, увидел
 * похожие» (сценарий Pinterest). Похожие подгружаются лениво при открытии:
 * они не нужны, пока человек не открыл конкретный снимок. Открытие снимка
 * также логируется (api.viewPhoto) — это один из сигналов персональной ленты.
 */
function PhotoLightbox({
  tile,
  onClose,
  onOpenSimilar,
  onToggleLike,
  onToggleFavorite,
}: {
  tile: Tile
  onClose: () => void
  onOpenSimilar: (tile: Tile) => void
  onToggleLike?: (tile: Tile) => void
  onToggleFavorite?: (tile: Tile) => void
}) {
  const [similar, setSimilar] = useState<Tile[] | null>(null)

  useEffect(() => {
    let alive = true
    setSimilar(null)
    api.viewPhoto(tile.databaseId, tile.photoId).catch(() => undefined)
    api
      .similar(tile.databaseId, tile.photoId)
      .then((result) => {
        if (!alive) return
        setSimilar(
          result.results.map((hit) => ({
            photoId: hit.photo_id,
            databaseId: tile.databaseId,
            thumbUrl: hit.thumb_url,
            fileUrl: hit.file_url,
            score: hit.score,
            caption: hit.caption,
            liked: hit.liked,
            favorited: hit.favorited,
            aiGenerated: hit.ai_generated,
          })),
        )
      })
      .catch(() => alive && setSimilar([]))
    return () => {
      alive = false
    }
  }, [tile.databaseId, tile.photoId])

  return (
    <div className="lightbox" role="dialog" aria-modal="true">
      <button type="button" className="lightbox__scrim" onClick={onClose} aria-label="Закрыть" />
      <div className="lightbox__panel">
        <button type="button" className="lightbox__close" onClick={onClose} aria-label="Закрыть">
          ×
        </button>
        <img className="lightbox__image" src={tile.fileUrl} alt="" />

        {(onToggleLike || onToggleFavorite) && (
          <div className="lightbox__marks">
            {onToggleLike && (
              <button
                type="button"
                className="card__mark"
                data-active={Boolean(tile.liked)}
                aria-label={tile.liked ? 'Убрать лайк' : 'Поставить лайк'}
                onClick={() => onToggleLike(tile)}
              >
                {tile.liked ? '♥' : '♡'}
              </button>
            )}
            {onToggleFavorite && (
              <button
                type="button"
                className="card__mark"
                data-active={Boolean(tile.favorited)}
                aria-label={tile.favorited ? 'Убрать из избранного' : 'В избранное'}
                onClick={() => onToggleFavorite(tile)}
              >
                {tile.favorited ? '★' : '☆'}
              </button>
            )}
          </div>
        )}

        <div className="lightbox__similar">
          <p className="eyebrow">Похожие</p>
          {similar === null ? (
            <p className="note">Ищем похожие…</p>
          ) : similar.length === 0 ? (
            <p className="note">Похожих не нашлось</p>
          ) : (
            <div className="lightbox__similar-row">
              {similar.map((hit) => (
                <button
                  key={hit.photoId}
                  type="button"
                  className="lightbox__similar-item"
                  onClick={() => onOpenSimilar(hit)}
                >
                  <img src={hit.thumbUrl} alt="" loading="lazy" />
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
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
