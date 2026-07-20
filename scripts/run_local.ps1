<#
.SYNOPSIS
    Запускает проект на этой машине, без Docker. Для демонстрации и разработки.

.DESCRIPTION
    Поднимает три части в отдельных окнах: API, фронтенд и (если задан токен)
    Telegram-бота. На ноутбуке инференс идёт заметно быстрее, чем в контейнере:
    Docker Desktop на Windows работает через виртуальную машину, и CPU там дороже.

    Данные те же самые — папка data/ общая с контейнерами. Поэтому одновременно
    поднимать оба варианта нельзя: два процесса писали бы в одни файлы индекса.
    Скрипт это проверяет и предупреждает.

.EXAMPLE
    .\scripts\run_local.ps1
    .\scripts\run_local.ps1 -NoBot        # без Telegram-бота
    .\scripts\run_local.ps1 -ApiOnly      # только API, без фронтенда
#>

param(
    [switch]$NoBot,
    [switch]$ApiOnly,
    [int]$ApiPort = 8000,
    [int]$UiPort = 5173
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }

Say ""
Say "  Запуск на этой машине (без Docker)" "Cyan"
Say "  ----------------------------------" "Cyan"

# --- проверки окружения -----------------------------------------------------

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Say "  Нет виртуального окружения .venv" "Red"
    Say "  Создайте его:  python -m venv .venv" "Yellow"
    Say "                 .venv\Scripts\python.exe -m pip install torch -r requirements-web.txt"
    exit 1
}

& $python -c "import fastapi, uvicorn" 2>$null
if (-not $?) {
    Say "  В .venv нет зависимостей веба" "Red"
    Say "  Поставьте:  .venv\Scripts\python.exe -m pip install -r requirements-web.txt" "Yellow"
    exit 1
}

# Контейнеры и локальный запуск делят папку data/ и токен бота. Одновременно
# они писали бы в одни и те же файлы индекса, а Telegram допускает только одного
# потребителя getUpdates на токен.
$running = ""
try { $running = (docker compose ps --services --filter "status=running" 2>$null) -join " " } catch {}
if ($running) {
    Say "  ВНИМАНИЕ: контейнеры уже работают: $running" "Yellow"
    Say "  Они пишут в ту же папку data/. Остановите их:  docker compose down" "Yellow"
    $answer = Read-Host "  Продолжить всё равно? (y/N)"
    if ($answer -ne "y") { exit 1 }
}

function Test-PortBusy($port) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $conn
}

if (Test-PortBusy $ApiPort) {
    Say "  Порт $ApiPort занят — освободите его или укажите другой: -ApiPort 8001" "Red"
    exit 1
}

# --- API --------------------------------------------------------------------

Say ""
Say "  API      -> http://127.0.0.1:$ApiPort" "Green"
Say "             первый запуск ждёт загрузки модели, это 15-30 секунд"
Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "web.app:app", "--port", "$ApiPort" `
    -WorkingDirectory $root

# --- фронтенд ---------------------------------------------------------------

if (-not $ApiOnly) {
    if (-not (Test-Path (Join-Path $root "web-ui\node_modules"))) {
        Say "  Ставлю зависимости фронтенда (один раз)..." "Yellow"
        npm --prefix web-ui install --no-audit --no-fund
    }
    Say "  Интерфейс -> http://localhost:$UiPort" "Green"
    Start-Process -FilePath "npm" `
        -ArgumentList "--prefix", "web-ui", "run", "dev" `
        -WorkingDirectory $root
}

# --- бот --------------------------------------------------------------------

if (-not $NoBot) {
    $envFile = Join-Path $root ".env"
    $hasToken = (Test-Path $envFile) -and ((Get-Content $envFile -Raw) -match "TELEGRAM_BOT_TOKEN\s*=\s*\S")
    if ($hasToken) {
        Say "  Бот      -> long polling через API на :$ApiPort" "Green"
        $env:API_URL = "http://127.0.0.1:$ApiPort"
        Start-Process -FilePath $python -ArgumentList "-m", "bot.bot" -WorkingDirectory $root
    }
    else {
        Say "  Бот не запущен: в .env нет TELEGRAM_BOT_TOKEN (это нормально)" "DarkGray"
    }
}

Say ""
Say "  Готово. Окна закрывать по Ctrl+C в каждом." "Cyan"
Say "  Данные лежат в data/ — те же, что видят контейнеры." "DarkGray"
Say ""
