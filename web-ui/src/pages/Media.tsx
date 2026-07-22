/**
 * «Фильмы и музыка» — подборка по темам недавних запросов, лайков, избранного
 * и просмотров (web/routers/media.py собирает темы и зовёт OMDb + Last.fm).
 * Каждая тема — то же самое ключевое слово, что привело к снимку в ленте,
 * поэтому вкладка ощущается продолжением того же самого интереса, а не
 * отдельным случайным разделом.
 */

import { useEffect, useState } from 'react'
import { ApiError, api, type Media as MediaData } from '../api'
import { Empty } from '../components'

export function Media() {
  const [data, setData] = useState<MediaData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .media()
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : 'Не удалось загрузить подборку'))
  }, [])

  if (!data) {
    return <div className="empty">{error ?? 'Подбираем фильмы и музыку…'}</div>
  }

  if (!data.enabled) {
    return (
      <Empty
        title="Вкладка пока выключена"
        hint="На сервере не настроены ключи каталогов фильмов и музыки."
      />
    )
  }

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="sakha-band" aria-hidden="true" />
      <header className="masthead">
        <div>
          <p className="eyebrow" style={{ marginTop: 0 }}>Фильмы и музыка</p>
          <h1 className="title" style={{ marginTop: 6 }}>По вашим темам</h1>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      {data.themes.length === 0 ? (
        <Empty
          title="Пока нечего подобрать"
          hint="Поищите что-нибудь или полайкайте снимки — темы для подборки появятся сами."
        />
      ) : (
        data.themes.map((theme) => (
          <section className="media-theme" key={theme.theme}>
            <h2 className="media-theme__title">{theme.theme}</h2>

            {theme.movies.length > 0 && (
              <div className="media-block">
                <p className="media-block__title">Фильмы</p>
                <div className="media-row">
                  {theme.movies.map((movie) => (
                    <a
                      key={movie.imdb_id}
                      className="media-card"
                      href={`https://www.imdb.com/title/${movie.imdb_id}/`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {movie.poster_url ? (
                        <img className="media-card__poster" src={movie.poster_url} alt="" loading="lazy" />
                      ) : (
                        <div className="media-card__poster media-card__poster--blank" aria-hidden="true" />
                      )}
                      <p className="media-card__title">{movie.title}</p>
                      <p className="media-card__meta">{movie.year}</p>
                    </a>
                  ))}
                </div>
              </div>
            )}

            {(theme.tracks.length > 0 || theme.artists.length > 0) && (
              <div className="media-block media-block--music">
                <p className="media-block__title">Музыка</p>

                {theme.tracks.length > 0 && (
                  <ul className="media-tracks">
                    {theme.tracks.map((track, i) => (
                      <li key={`${track.name}-${track.artist}-${i}`}>
                        <span className="media-tracks__name">{track.name}</span>
                        <span className="media-tracks__artist">{track.artist}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {theme.artists.length > 0 && (
                  <div className="media-row">
                    {theme.artists.map((artist) => (
                      <div className="media-artist" key={artist.name}>
                        {artist.image_url ? (
                          <img className="media-artist__image" src={artist.image_url} alt="" loading="lazy" />
                        ) : (
                          <span className="media-artist__image media-artist__image--fallback" aria-hidden="true">
                            {artist.name.trim().charAt(0).toUpperCase() || '?'}
                          </span>
                        )}
                        <span>{artist.name}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </section>
        ))
      )}
    </div>
  )
}
