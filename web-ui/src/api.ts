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
  avatar_url: string | null
}

export interface Database {
  id: string
  name: string
  photos_count: number
  photos_bytes: number
  index_bytes: number
  total_bytes: number
  has_captions: boolean
  /** сколько снимков уже размечено подписями — фоновая разметка идёт постепенно */
  captions_count: number
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
  caption: string
  liked: boolean
  favorited: boolean
  ai_generated: boolean
}

export interface PhotoPage {
  total: number
  offset: number
  items: Photo[]
}

export interface CaptionResult {
  photo_id: string
  caption: string
  /** попала ли подпись в поисковый индекс (иначе сохранена только как текст) */
  indexed: boolean
}

export interface SearchHit {
  photo_id: string
  /** нужен ленте — она собирает снимки сразу из нескольких баз */
  database_id: string | null
  score: number
  thumb_url: string
  file_url: string
  caption: string
  liked: boolean
  favorited: boolean
  /** снимок сгенерирован YandexART, а не найден в базе (поиск ничего не нашёл) */
  ai_generated: boolean
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

export interface ProfilePhoto {
  database_id: string
  database_name: string
  database_kind: 'personal' | 'chat' | 'demo'
  photo_id: string
  marked_at: string
  thumb_url: string
  file_url: string
}

export interface Profile {
  user: User
  liked: ProfilePhoto[]
  favorited: ProfilePhoto[]
}

export interface Movie {
  title: string
  year: string
  imdb_id: string
  poster_url: string | null
}

export interface Track {
  name: string
  artist: string
  url: string
}

export interface Artist {
  name: string
  url: string
  image_url: string | null
}

export interface MediaTheme {
  theme: string
  movies: Movie[]
  tracks: Track[]
  artists: Artist[]
}

export interface Media {
  /** false — ни один ключ каталога не настроен на сервере, вкладку показывать незачем */
  enabled: boolean
  themes: MediaTheme[]
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
  avatarUrl: () => `/api/me/avatar?t=${Date.now()}`,
  uploadAvatar: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<User>('/me/avatar', { method: 'POST', body: form })
  },
  deleteAvatar: () => request<User>('/me/avatar', { method: 'DELETE' }),
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
  setCaption: (id: string, photoId: string, caption: string) =>
    json<CaptionResult>(`/databases/${id}/photos/${photoId}/caption`, 'PUT', { caption }),

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

  generate: (id: string, query: string) =>
    json<SearchResult>(`/databases/${id}/search/generate`, 'POST', { query }),

  searchImage: (id: string, file: File, top_k = 12) => {
    const form = new FormData()
    form.append('file', file)
    return request<SearchResult>(`/databases/${id}/search/image?top_k=${top_k}`, {
      method: 'POST',
      body: form,
    })
  },

  similar: (id: string, photoId: string, top_k = 12) =>
    request<SearchResult>(`/databases/${id}/search/similar/${photoId}?top_k=${top_k}`),

  photo: (id: string, photoId: string) => request<Photo>(`/databases/${id}/photos/${photoId}/info`),

  viewPhoto: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}/view`, { method: 'POST' }),

  feed: () => request<SearchResult>('/feed'),
  media: () => request<Media>('/media'),

  job: (jobId: string) => request<Job>(`/jobs/${jobId}`),
  jobs: (databaseId: string) => request<Job[]>(`/jobs?database_id=${databaseId}`),
  cancelJob: (jobId: string) => json<Job>(`/jobs/${jobId}/cancel`, 'POST'),

  like: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}/like`, { method: 'PUT' }),
  unlike: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}/like`, { method: 'DELETE' }),
  favorite: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}/favorite`, { method: 'PUT' }),
  unfavorite: (id: string, photoId: string) =>
    request<void>(`/databases/${id}/photos/${photoId}/favorite`, { method: 'DELETE' }),
  profile: () => request<Profile>('/profile'),

  exportUrl: (id: string) => `/api/databases/${id}/export.zip`,
  thumbUrl: (id: string, photoId: string) => `/api/databases/${id}/photos/${photoId}/thumb`,
  fileUrl: (id: string, photoId: string) => `/api/databases/${id}/photos/${photoId}/file`,
}
