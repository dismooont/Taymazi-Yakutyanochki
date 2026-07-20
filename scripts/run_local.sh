#!/usr/bin/env bash
# Запуск проекта на этой машине, без Docker (Linux/macOS).
# Версия для Windows: scripts/run_local.ps1
#
# Данные — папка data/ — общие с контейнерами, поэтому одновременно поднимать
# оба варианта нельзя: два процесса писали бы в одни файлы индекса.

set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Нет виртуального окружения ($PYTHON)."
    echo "Создайте:  python3 -m venv .venv && .venv/bin/pip install torch -r requirements-web.txt"
    exit 1
fi

if docker compose ps --services --filter status=running 2>/dev/null | grep -q .; then
    echo "ВНИМАНИЕ: контейнеры работают и пишут в ту же папку data/."
    echo "Остановите их:  docker compose down"
    read -r -p "Продолжить всё равно? (y/N) " answer
    [ "$answer" = "y" ] || exit 1
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "API      -> http://127.0.0.1:$API_PORT  (первый запуск ждёт модель, 15-30 с)"
"$PYTHON" -m uvicorn web.app:app --port "$API_PORT" &
pids+=($!)

if [ ! -d web-ui/node_modules ]; then
    echo "Ставлю зависимости фронтенда (один раз)..."
    npm --prefix web-ui install --no-audit --no-fund
fi
echo "Интерфейс -> http://localhost:$UI_PORT"
npm --prefix web-ui run dev &
pids+=($!)

if grep -qE '^\s*TELEGRAM_BOT_TOKEN\s*=\s*\S' .env 2>/dev/null; then
    echo "Бот      -> long polling через API на :$API_PORT"
    API_URL="http://127.0.0.1:$API_PORT" "$PYTHON" -m bot.bot &
    pids+=($!)
else
    echo "Бот не запущен: в .env нет TELEGRAM_BOT_TOKEN (это нормально)"
fi

echo
echo "Готово. Остановить всё — Ctrl+C."
wait
