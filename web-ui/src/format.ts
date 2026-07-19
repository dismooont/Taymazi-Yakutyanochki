/** Форматирование величин, которые пользователь видит в интерфейсе. */

/**
 * Объём в привычных единицах. Переход на следующую единицу — с 1024,
 * дробная часть только там, где она что-то значит: «870 КБ», но «2,1 ГБ».
 */
export function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 МБ'
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  const digits = value < 10 && unit >= 2 ? 1 : 0
  return `${value.toFixed(digits).replace('.', ',')} ${units[unit]}`
}

/** «1 240 фото» — с правильным окончанием и неразрывными разрядами. */
export function formatPhotos(count: number): string {
  const grouped = count.toLocaleString('ru-RU').replace(/\s/g, ' ')
  const rest = count % 100
  const last = count % 10
  if (rest >= 11 && rest <= 14) return `${grouped} фотографий`
  if (last === 1) return `${grouped} фотография`
  if (last >= 2 && last <= 4) return `${grouped} фотографии`
  return `${grouped} фотографий`
}

export function formatDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' })
}

/** Косинусная близость CLIP: показываем как есть, четырьмя знаками. */
export function formatScore(score: number): string {
  return score.toFixed(4).replace('.', ',')
}
