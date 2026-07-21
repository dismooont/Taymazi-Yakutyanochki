import { useEffect, useState } from 'react'
import { Link, Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { api, type User } from './api'
import { Auth } from './pages/Auth'
import { Databases } from './pages/Databases'
import { Feed } from './pages/Feed'
import { Profile } from './pages/Profile'
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
        <nav className="topbar__nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `topbar__link${isActive ? ' active' : ''}`}
          >
            Лента
          </NavLink>
          <NavLink
            to="/databases"
            className={({ isActive }) => `topbar__link${isActive ? ' active' : ''}`}
          >
            Мои базы
          </NavLink>
        </nav>
        <span className="topbar__spacer" />
        <Link to="/profile" className="note" style={{ textDecoration: 'none' }}>
          {user.display_name}
        </Link>
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
          <Route path="/" element={<Feed />} />
          <Route path="/databases" element={<Databases />} />
          <Route path="/db/:id" element={<Workspace />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
