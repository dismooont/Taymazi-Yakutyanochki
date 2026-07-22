import { useEffect, useState } from 'react'
import { Link, Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { api, type User } from './api'
import { MammothMark } from './AuroraScene'
import { Avatar } from './components'
import { Auth } from './pages/Auth'
import { Databases } from './pages/Databases'
import { Feed } from './pages/Feed'
import { Media } from './pages/Media'
import { Photo } from './pages/Photo'
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
        <Link to="/" className="topbar__mark">
          <MammothMark className="topbar__mark-mammoth" />
          <span className="topbar__mark-text">Поиск по архиву</span>
        </Link>
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
          <NavLink
            to="/media"
            className={({ isActive }) => `topbar__link${isActive ? ' active' : ''}`}
          >
            Фильмы и музыка
          </NavLink>
        </nav>
        <span className="topbar__spacer" />
        <button
          type="button"
          className="btn btn--quiet topbar__logout"
          onClick={() => api.logout().finally(() => setUser(null))}
        >
          Выйти
        </button>
        <Link to="/profile" className="topbar__profile" title={user.display_name}>
          <Avatar user={user} size={36} />
        </Link>
      </header>

      <main className="page">
        <Routes>
          <Route path="/" element={<Feed />} />
          <Route path="/databases" element={<Databases />} />
          <Route path="/media" element={<Media />} />
          <Route path="/db/:id" element={<Workspace />} />
          <Route path="/db/:id/photo/:photoId" element={<Photo />} />
          <Route path="/profile" element={<Profile onUserUpdate={setUser} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
