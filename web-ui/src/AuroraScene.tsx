/**
 * Якутская ночная сцена — фон экрана входа: полярное сияние, звёзды, снег,
 * гряда гор и силуэт мамонта. Всё анимировано на CSS (см. styles.css, секция
 * «якутская ночь»), поэтому здесь только разметка и разовая генерация
 * случайных снежинок/звёзд — без таймеров и перерисовок в рантайме.
 *
 * Раньше тему Саха держала одна большая полупрозрачная фотография мамонта на
 * фоне; выглядело тяжело и грязно. Сцена собрана из лёгких примитивов (SVG +
 * градиенты), поэтому читается чисто и живёт сама по себе.
 */

import { useMemo } from 'react'

type Flake = { left: number; size: number; duration: number; delay: number; drift: number }
type Star = { left: number; top: number; size: number; delay: number; duration: number }

function useFlakes(count: number): Flake[] {
  return useMemo(
    () =>
      Array.from({ length: count }, () => ({
        left: Math.random() * 100,
        size: 1.5 + Math.random() * 2.5,
        duration: 7 + Math.random() * 9,
        delay: -Math.random() * 16, // отрицательная задержка — снег уже идёт при загрузке, а не «сыплется с потолка»
        drift: (Math.random() - 0.5) * 40,
      })),
    [count],
  )
}

function useStars(count: number): Star[] {
  return useMemo(
    () =>
      Array.from({ length: count }, () => ({
        left: Math.random() * 100,
        top: Math.random() * 55, // только в верхней части неба, над горами
        size: 1 + Math.random() * 1.6,
        // у каждой звезды своя длинная фаза и период — иначе всё мерцает
        // синхронно и «мигает»; вразнобой и медленно это уже тихое дыхание.
        delay: -Math.random() * 12,
        duration: 6 + Math.random() * 8,
      })),
    [count],
  )
}

/**
 * Силуэт мамонта — общая фигура для большой сцены и для крохотного знака в
 * шапке (MammothMark). Только пути, без <svg>-обёртки, чтобы вызывающий сам
 * задал размер и viewBox 0 0 260 180.
 */
function MammothPaths() {
  return (
    <>
      {/* ноги */}
      <rect x="70" y="112" width="24" height="46" rx="10" />
      <rect x="104" y="114" width="22" height="44" rx="10" />
      <rect x="150" y="112" width="24" height="46" rx="10" />
      <rect x="182" y="114" width="22" height="44" rx="10" />
      {/* туловище с горбом-загривком и куполом головы */}
      <path
        d="M52 150
           C44 124, 44 92, 64 74
           C82 58, 108 52, 128 56
           C144 44, 168 44, 186 58
           C198 50, 216 52, 226 68
           C234 80, 234 96, 227 108
           C222 116, 214 120, 204 122
           C176 126, 120 130, 92 128
           C74 127, 60 132, 54 148
           Z"
      />
      {/* хобот, спускается и заворачивается */}
      <path
        d="M222 106
           C233 112, 240 124, 239 138
           C238 150, 231 162, 222 165
           C217 167, 212 165, 213 159
           C214 152, 221 150, 223 143
           C225 136, 222 129, 216 126
           C211 123, 208 114, 212 108
           Z"
      />
      {/* бивень */}
      <path
        d="M214 118
           C200 128, 190 146, 196 162
           C197 166, 201 166, 201 161
           C199 148, 208 133, 222 124
           Z"
      />
    </>
  )
}

/** Компактный знак-мамонт для шапки (заменяет прежнюю синюю точку логотипа). */
export function MammothMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 260 180" width="26" height="18" aria-hidden="true">
      <g fill="currentColor">
        <MammothPaths />
      </g>
    </svg>
  )
}

export function AuroraScene() {
  const flakes = useFlakes(34)
  const stars = useStars(40)

  return (
    <div className="scene" aria-hidden="true">
      <div className="scene__sky" />
      {/* сплошной слой поверх неба, что бесконечно и плавно переливается по
          всей ширине — та самая «перелив», а не мигание отдельных точек */}
      <div className="scene__flow" />
      <div className="scene__moon" />

      <div className="scene__stars">
        {stars.map((s, i) => (
          <span
            key={i}
            className="scene__star"
            style={{
              left: `${s.left}%`,
              top: `${s.top}%`,
              width: s.size,
              height: s.size,
              animationDelay: `${s.delay}s`,
              animationDuration: `${s.duration}s`,
            }}
          />
        ))}
      </div>

      {/* «Занавесы» сияния разной ширины и оттенка — размытые, медленно текут
          и колышутся; за счёт разных длительностей не повторяются синхронно. */}
      <div className="scene__aurora scene__aurora--1" />
      <div className="scene__aurora scene__aurora--2" />
      <div className="scene__aurora scene__aurora--3" />
      <div className="scene__aurora scene__aurora--4" />

      <div className="scene__snow">
        {flakes.map((f, i) => (
          <span
            key={i}
            className="scene__flake"
            style={
              {
                left: `${f.left}%`,
                width: f.size,
                height: f.size,
                animationDuration: `${f.duration}s`,
                animationDelay: `${f.delay}s`,
                '--drift': `${f.drift}px`,
              } as React.CSSProperties
            }
          />
        ))}
      </div>

      <svg className="scene__ridges" viewBox="0 0 1440 320" preserveAspectRatio="none">
        {/* дальняя гряда — светлее и выше, создаёт глубину */}
        <path
          className="scene__ridge scene__ridge--far"
          d="M0,220 L180,120 L320,190 L470,90 L640,180 L800,110 L980,200 L1150,120 L1310,190 L1440,140 L1440,320 L0,320 Z"
        />
        {/* ближняя гряда — темнее, со снежными шапками поверх пиков */}
        <path
          className="scene__ridge scene__ridge--near"
          d="M0,300 L120,210 L280,280 L430,190 L560,260 L720,180 L900,270 L1080,200 L1260,275 L1440,210 L1440,320 L0,320 Z"
        />
        <path
          className="scene__snowcap"
          d="M430,190 L470,218 L560,225 L595,255 L500,250 L465,240 L440,225 Z"
        />
        <path
          className="scene__snowcap"
          d="M720,180 L760,210 L860,220 L900,270 L790,255 L750,235 L730,215 Z"
        />
      </svg>

      {/* Мамонт стоит на переднем плане поверх ближней гряды, чуть покачиваясь. */}
      <svg className="scene__mammoth" viewBox="0 0 260 180" preserveAspectRatio="xMidYMax meet">
        <g className="scene__mammoth-body">
          <MammothPaths />
        </g>
      </svg>

      {/* мягкое осветление снизу, чтобы белая карточка входа не «висела» на резкой границе */}
      <div className="scene__fade" />
    </div>
  )
}
