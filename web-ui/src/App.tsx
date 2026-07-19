import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { api, type User } from './api'
import { Auth } from './pages/Auth'
import { Databases } from './pages/Databases'
import { Workspace } from './pages/Workspace'

export function App() {
  // undefined — ещё не знаем, есть ли сессия; null — точно не вошли
  const [user, setUser] = useState<User | null | undefined>(undefined)

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
  }, [])

  if (user === undefined) {
    return <div className="empty" style={{ margin: 80 }}>Проверяем сессию…</div>
  }

  if (user === null) {
    return <Auth onAuth={setUser} />
  }

  return (
    <div className="shell">
      <header className="topbar">
        <span className="topbar__mark">Поиск по архиву</span>
        <span className="topbar__spacer" />
        <span className="note">{user.display_name}</span>
        <button
          type="button"
          className="btn btn--quiet"
          onClick={() => api.logout().finally(() => setUser(null))}
        >
          Выйти
        </button>
      </header>

      <main className="page">
        <Routes>
          <Route path="/" element={<Databases />} />
          <Route path="/db/:id" element={<Workspace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
