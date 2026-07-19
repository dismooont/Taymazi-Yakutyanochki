/**
 * Выбор базы: загрузить архив, создать пустую или вернуться к уже существующей.
 * Это первый экран после входа — то самое «окно с выбором» из сценария.
 */

import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError, api, type Database, type Quota } from '../api'
import { Dropzone, Empty, JobProgress, Toast, useJob } from '../components'
import { formatBytes, formatDate, formatPhotos } from '../format'

export function Databases() {
  const navigate = useNavigate()
  const [databases, setDatabases] = useState<Database[] | null>(null)
  const [quota, setQuota] = useState<Quota | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const [list, limits] = await Promise.all([api.databases(), api.quota()])
    setDatabases(list)
    setQuota(limits)
  }, [])

  useEffect(() => {
    refresh().catch(() => setError('Не удалось загрузить список баз'))
  }, [refresh])

  const job = useJob(jobId, () => {
    refresh().catch(() => undefined)
    setToast('Архив обработан')
  })

  const askName = (fallback: string) => {
    const name = window.prompt('Название базы', fallback)
    return name?.trim() || null
  }

  const createEmpty = async () => {
    const name = askName('Новая база')
    if (!name) return
    setBusy(true)
    setError(null)
    try {
      const database = await api.createDatabase(name)
      navigate(`/db/${database.id}`)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось создать базу')
    } finally {
      setBusy(false)
    }
  }

  const importArchive = async (files: File[]) => {
    const file = files[0]
    if (!file) return
    const name = askName(file.name.replace(/\.zip$/i, ''))
    if (!name) return

    setBusy(true)
    setError(null)
    let created: Database | null = null
    try {
      created = await api.createDatabase(name)
      const result = await api.importArchive(created.id, file)
      setJobId(result.job_id)
      await refresh()
    } catch (e) {
      // база уже создана, но архив не принят — не оставляем пустышку в списке
      if (created) await api.deleteDatabase(created.id).catch(() => undefined)
      setError(e instanceof ApiError ? e.message : 'Не удалось загрузить архив')
    } finally {
      setBusy(false)
    }
  }

  const remove = async (database: Database) => {
    const sure = window.confirm(
      `Удалить базу «${database.name}» и все ${database.photos_count} снимков? Это необратимо.`,
    )
    if (!sure) return
    await api.deleteDatabase(database.id)
    setToast(`База «${database.name}» удалена`)
    await refresh()
  }

  return (
    <div className="stack" style={{ gap: 36 }}>
      <div>
        <p className="eyebrow">Ваши базы</p>
        <h1 className="title">С чего начнём</h1>
      </div>

      <div className="starters">
        <Dropzone accept=".zip,application/zip" onFiles={importArchive}>
          <p className="eyebrow">Архив уже есть</p>
          <p className="starter__title">Загрузить архив</p>
          <p className="note">
            Перетащите zip с фотографиями или нажмите, чтобы выбрать файл. Снимки
            проиндексируются в фоне.
          </p>
        </Dropzone>

        <button type="button" className="starter" onClick={createEmpty} disabled={busy}>
          <p className="eyebrow">Архива нет</p>
          <p className="starter__title">Создать пустую базу</p>
          <p className="note">Начните с нуля и добавляйте снимки по мере надобности.</p>
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {job && <JobProgress job={job} onCancel={() => api.cancelJob(job.id).catch(() => undefined)} />}

      <div className="stack">
        <div className="row">
          <p className="eyebrow" style={{ margin: 0 }}>
            Готовые базы
          </p>
          <span className="topbar__spacer" />
          {quota && (
            <span className="mono" style={{ color: 'var(--muted)' }}>
              {quota.databases_used}/{quota.databases_limit} баз · {formatBytes(quota.bytes_used)} из{' '}
              {formatBytes(quota.bytes_limit)}
            </span>
          )}
        </div>

        {databases === null && <div className="empty">Загружаем…</div>}

        {databases?.length === 0 && (
          <Empty
            title="Пока ни одной базы"
            hint="Загрузите архив или создайте пустую базу — оба варианта выше."
          />
        )}

        {databases?.map((database) => (
          <div className="entry" key={database.id}>
            <div className="entry__main">
              {/* превью показывает, что внутри: список из одних названий не даёт
                  вспомнить, какая база какая */}
              {database.preview.length > 0 && (
                <div className="entry__preview" aria-hidden="true">
                  {database.preview.map((photoId) => (
                    <img key={photoId} src={api.thumbUrl(database.id, photoId)} alt="" loading="lazy" />
                  ))}
                </div>
              )}
              <div>
                <p className="entry__name">
                  {database.name}
                  {database.kind !== 'personal' && (
                    <span className="tag">
                      {database.kind === 'chat' ? 'чат' : 'только чтение'}
                    </span>
                  )}
                </p>
                <p className="entry__meta mono">
                  {formatPhotos(database.photos_count)} · {formatBytes(database.total_bytes)} ·
                  создана {formatDate(database.created_at)}
                </p>
              </div>
            </div>
            <div className="row">
              <button
                type="button"
                className="btn"
                onClick={() => navigate(`/db/${database.id}`)}
              >
                Открыть
              </button>
              {/* демо-база общая: предлагать её удаление означало бы обещать
                  действие, которое сервер всё равно отклонит */}
              {!database.read_only && (
                <button
                  type="button"
                  className="btn btn--quiet btn--danger"
                  onClick={() => remove(database)}
                  aria-label={`Удалить базу ${database.name}`}
                >
                  Удалить
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {toast && <Toast text={toast} onDone={() => setToast(null)} />}
    </div>
  )
}
