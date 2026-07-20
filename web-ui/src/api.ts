/**
 * Клиент API.
 *
 * Все запросы идут с cookie сессии (credentials) и заголовком X-Requested-With —
 * без него бэкенд отклоняет любые изменяющие запросы (защита от CSRF).
 */

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

export interface User {
  id: string
  login: string | null
  display_name: string
  has_password: boolean
  has_telegram: boolean
}

export interface Database {
  id: string
  name: string
  photos_count: number
  photos_bytes: number
  index_bytes: number
  total_bytes: number
  has_captions: boolean
  status: string
  /** personal — своя база, chat — база Telegram-чата, demo — общая витрина */
  kind: 'personal' | 'chat' | 'demo'
  read_only: boolean
  /** photo_id первых снимков — для превью в списке баз */
  preview: string[]
  created_at: string
  updated_at: string
}

export interface Photo {
  photo_id: string
  bytes: number
  added_at: string
}

export interface PhotoPage {
  total: number
  offset: number
  items: Photo[]
}

export interface SearchHit {
  photo_id: string
  score: number
  thumb_url: string
  file_url: string
  caption: string
}

export interface CaptionHit {
  photo_id: string
  score: number
  caption: string
}

export interface SearchResult {
  used_query: string | null
  results: SearchHit[]
  captions: CaptionHit[]
  /** Выдача собрана слиянием с поиском по подписям: оценка тогда не косинус. */
  fused: boolean
}

export interface Job {
  id: string
  kind: string
  status: 'queued' | 'running' | 'done' | 'error'
  database_id: string | null
  progress_done: number
  progress_total: number
  queue_position: number
  message: string | null
  created_at: string
  finished_at: string | null
}

export interface AddPhotosResult {
  job_id: string | null
  added: number
  skipped: [string, string][]
}

export interface Quota {
  databases_used: number
  databases_limit: number
  bytes_used: number
  bytes_limit: number
  photos_per_database_limit: number
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    credentials: 'include',
    ...init,
    headers: { 'X-Requested-With': 'XMLHttpRequest', ...(init.headers ?? {}) },
  })

  if (!response.ok) {
    let detail = `Ошибка ${response.status}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
      // FastAPI отдаёт ошибки валидации массивом — показываем первую
      else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail
    } catch {
      /* тело не JSON — остаётся код ответа */
    }
    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function json<T>(path: string, method: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export interface PublicConfig {
  registration_open: boolean
  telegram_auth: boolean
  telegram_bot: string | null
}

/** Данные, которые присылает виджет Telegram. Проверяются на сервере по подписи. */
export type TelegramPayload = Record<string, string | number>

export const api = {
  config: () => request<PublicConfig>('/config'),
  me: () => request<User>('/me'),
  telegramAuth: (payload: TelegramPayload) => json<User>('/auth/telegram', 'POST', payload),
  unlinkTelegram: () => request<User>('/me/identities/telegram', { method: 'DELETE' }),
  register: (login: string, password: string, display_name?: string) =>
    json<User>('/auth/register', 'POST', { login, password, display_name }),
  login: (login: string, password: string) => json<User>('/auth/login', 'POST', { login, password }),
  logout: () => request<void>('/auth/logout', { method: 'POST' }),

  quota: () => request<Quota>('/quota'),
  databases: () => request<Database[]>('/databases'),
  database: (id: string) => request<Database>(`/databases/${id}`),
  stats: (id: string) => request<Database>(`/databases/${id}/stats`),
  createDatabase: (name: string) => json<Database>('/databases', 'POST', { name }),
  renameDatabase: (id: string, name: string) => json<Database>(`/databases/${id}`, 'PATCH', { name }),
  deleteDatabase: (id: string) => request<void>(`/databases/${id}`, { method: 'DELETE' }),

  photos: (id: string, offset = 0, limit = 60) =>
    request<PhotoPage>(`/databases/${id}/photos?offset=${offset}&limit=${limit}`),
  deletePhoto: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}`, { method: 'DELETE' }),

  addPhotos: (id: string, files: File[]) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return request<AddPhotosResult>(`/databases/${id}/photos`, { method: 'POST', body: form })
  },

  importArchive: (id: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<AddPhotosResult>(`/databases/${id}/import`, { method: 'POST', body: form })
  },

  searchText: (id: string, query: string, top_k = 12, translate = true) =>
    json<SearchResult>(`/databases/${id}/search/text`, 'POST', { query, top_k, translate }),

  searchImage: (id: string, file: File, top_k = 12) => {
    const form = new FormData()
    form.append('file', file)
    return request<SearchResult>(`/databases/${id}/search/image?top_k=${top_k}`, {
      method: 'POST',
      body: form,
    })
  },

  job: (jobId: string) => request<Job>(`/jobs/${jobId}`),
  jobs: (databaseId: string) => request<Job[]>(`/jobs?database_id=${databaseId}`),
  cancelJob: (jobId: string) => json<Job>(`/jobs/${jobId}/cancel`, 'POST'),

  exportUrl: (id: string) => `/api/databases/${id}/export.zip`,
  thumbUrl: (id: string, photoId: string) => `/api/databases/${id}/photos/${photoId}/thumb`,
  fileUrl: (id: string, photoId: string) => `/api/databases/${id}/photos/${photoId}/file`,
}
