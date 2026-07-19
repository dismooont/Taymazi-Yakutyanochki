/**
 * Вход и регистрация.
 *
 * Про восстановление пароля сказано прямо: самообслуживания нет, потому что для
 * писем со сбросом нужен почтовый сервер. Умолчать об этом означало бы оставить
 * человека выяснять это в тот момент, когда пароль уже забыт.
 */

import { useState } from 'react'
import { ApiError, api, type User } from '../api'

type Mode = 'login' | 'register'

export function Auth({ onAuth }: { onAuth: (user: User) => void }) {
  const [mode, setMode] = useState<Mode>('login')
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)

    if (mode === 'register' && password !== confirm) {
      setError('Пароли не совпадают')
      return
    }

    setBusy(true)
    try {
      const user =
        mode === 'login'
          ? await api.login(login, password)
          : await api.register(login, password)
      onAuth(user)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Не удалось связаться с сервером')
    } finally {
      setBusy(false)
    }
  }

  const switchTo = (next: Mode) => {
    setMode(next)
    setError(null)
    setConfirm('')
  }

  return (
    <div className="auth stack">
      <div>
        <p className="eyebrow">Семантический поиск по фотоархиву</p>
        <h1 className="title">Найдите снимок словами</h1>
      </div>

      <div className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'login'}
          className="tabs__item"
          onClick={() => switchTo('login')}
        >
          Вход
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'register'}
          className="tabs__item"
          onClick={() => switchTo('register')}
        >
          Регистрация
        </button>
      </div>

      <form className="stack" onSubmit={submit}>
        <div>
          <label className="label" htmlFor="login">
            Логин
          </label>
          <input
            id="login"
            className="field"
            value={login}
            autoComplete="username"
            onChange={(event) => setLogin(event.target.value)}
            required
          />
        </div>

        <div>
          <label className="label" htmlFor="password">
            Пароль
          </label>
          <input
            id="password"
            className="field"
            type="password"
            value={password}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
          {mode === 'register' && <p className="note">Не короче 10 символов</p>}
        </div>

        {mode === 'register' && (
          <div>
            <label className="label" htmlFor="confirm">
              Пароль ещё раз
            </label>
            <input
              id="confirm"
              className="field"
              type="password"
              value={confirm}
              autoComplete="new-password"
              onChange={(event) => setConfirm(event.target.value)}
              required
            />
          </div>
        )}

        {error && <p className="error">{error}</p>}

        <button type="submit" className="btn btn--primary" disabled={busy}>
          {busy ? 'Подождите' : mode === 'login' ? 'Войти' : 'Создать аккаунт'}
        </button>
      </form>

      <p className="note">
        Пароль восстановить не получится: почтовый сервер для писем со сбросом не
        подключён. Запишите пароль в надёжном месте.
      </p>
    </div>
  )
}
