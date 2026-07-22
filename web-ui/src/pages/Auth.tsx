/**
 * Вход и регистрация.
 *
 * Про восстановление пароля сказано прямо: самообслуживания нет, потому что для
 * писем со сбросом нужен почтовый сервер. Умолчать об этом означало бы оставить
 * человека выяснять это в тот момент, когда пароль уже забыт.
 */

import { useEffect, useRef, useState } from 'react'
import { ApiError, api, type PublicConfig, type TelegramPayload, type User } from '../api'
import { AuroraScene } from '../AuroraScene'

type Mode = 'login' | 'register'

declare global {
  interface Window {
    onTelegramAuth?: (user: TelegramPayload) => void
  }
}

/**
 * Кнопка входа через Telegram.
 *
 * Виджет — это скрипт с сайта Telegram, который сам рисует кнопку и вызывает
 * глобальную функцию с подписанными данными. Их нельзя принимать на веру: подпись
 * проверяет сервер по токену бота.
 */
function TelegramButton({ bot, onAuth }: { bot: string; onAuth: (p: TelegramPayload) => void }) {
  const holder = useRef<HTMLDivElement>(null)

  useEffect(() => {
    window.onTelegramAuth = onAuth
    const script = document.createElement('script')
    script.src = 'https://telegram.org/js/telegram-widget.js?22'
    script.async = true
    script.setAttribute('data-telegram-login', bot)
    script.setAttribute('data-size', 'large')
    script.setAttribute('data-radius', '3')
    script.setAttribute('data-onauth', 'onTelegramAuth(user)')
    holder.current?.appendChild(script)

    const node = holder.current
    return () => {
      if (node) node.innerHTML = ''
      delete window.onTelegramAuth
    }
  }, [bot, onAuth])

  return <div ref={holder} />
}

export function Auth({ onAuth }: { onAuth: (user: User) => void }) {
  const [mode, setMode] = useState<Mode>('login')
  const [login, setLogin] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [config, setConfig] = useState<PublicConfig | null>(null)

  useEffect(() => {
    api.config().then(setConfig).catch(() => undefined)
  }, [])

  const enterWithTelegram = async (payload: TelegramPayload) => {
    setError(null)
    try {
      onAuth(await api.telegramAuth(payload))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Telegram не ответил')
    }
  }

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
    <div className="auth-page">
      <AuroraScene />
      <div className="auth stack">
        <div className="sakha-band" aria-hidden="true" />
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

      {config?.telegram_auth && config.telegram_bot && (
        <div className="stack" style={{ gap: 10 }}>
          <div className="divider">
            <span>или</span>
          </div>
          <TelegramButton bot={config.telegram_bot} onAuth={enterWithTelegram} />
        </div>
      )}

        <p className="note">
          {config?.telegram_auth
            ? 'Пароль восстановить не получится: почтовый сервер для писем со сбросом не подключён. Привяжите Telegram — через него можно будет войти.'
            : 'Пароль восстановить не получится: почтовый сервер для писем со сбросом не подключён. Запишите пароль в надёжном месте.'}
        </p>
      </div>
    </div>
  )
}
