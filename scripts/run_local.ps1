<#
.SYNOPSIS
    Запускает проект на этой машине, без Docker. Для демонстрации и разработки.

.DESCRIPTION
    Поднимает три части в отдельных окнах: API, фронтенд и (если задан токен)
    Telegram-бота. Добавление фотографий здесь примерно в 11 раз быстрее, чем
    в контейнере (0,6 с против 6,8 с на снимок): папка data/ подключена к контейнеру
    как bind mount с Windows, и запись мелких файлов через виртуальную машину дорога.
    Поиск при этом почти не отличается — 82 мс против 119 мс.

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

$npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $npm -and -not $ApiOnly) {
    Say "  Не найден npm.cmd — установите Node.js или запускайте с ключом -ApiOnly" "Red"
    exit 1
}

# Порт может быть занят нашим же API с прошлого запуска — тогда второй поднимать
# не нужно и нельзя. Отличаем свой живой API от чужого процесса по /api/health:
# просто «порт занят -> выход» мешало бы дозапустить недостающие части.
$apiAlready = $false
if (Test-PortBusy $ApiPort) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5
        $apiAlready = ($health.status -eq "ok")
    }
    catch { $apiAlready = $false }

    if (-not $apiAlready) {
        Say "  Порт $ApiPort занят чужим процессом — освободите его или укажите другой: -ApiPort 8001" "Red"
        exit 1
    }
}

# --- API --------------------------------------------------------------------

Say ""
if ($apiAlready) {
    Say "  API      -> http://127.0.0.1:$ApiPort (уже работает, не поднимаю второй)" "Green"
}
else {
    Say "  API      -> http://127.0.0.1:$ApiPort" "Green"
    Say "             первый запуск ждёт загрузки модели, это 15-30 секунд"
    Start-Process -FilePath $python `
        -ArgumentList "-m", "uvicorn", "web.app:app", "--port", "$ApiPort" `
        -WorkingDirectory $root
}

# --- фронтенд ---------------------------------------------------------------

if (-not $ApiOnly -and (Test-PortBusy $UiPort)) {
    # Vite при занятом порте молча берёт следующий (5174), и человек продолжает
    # смотреть на старую вкладку, не понимая, почему правок не видно.
    Say "  Интерфейс -> http://localhost:$UiPort (уже работает, не поднимаю второй)" "Green"
}
elseif (-not $ApiOnly) {
    if (-not (Test-Path (Join-Path $root "web-ui\node_modules"))) {
        Say "  Ставлю зависимости фронтенда (один раз)..." "Yellow"
        & $npm --prefix web-ui install --no-audit --no-fund
    }
    Say "  Интерфейс -> http://localhost:$UiPort" "Green"
    # Именно npm.cmd, а не npm: в PATH первым лежит npm.ps1, и Start-Process его
    # запустить не может — "%1 is not a valid Win32 application". Окно фронтенда
    # при этом молча не открывалось, а скрипт рапортовал об успехе.
    Start-Process -FilePath $npm -ArgumentList "--prefix", "web-ui", "run", "dev" -WorkingDirectory $root

    # Ждём, пока Vite действительно начнёт слушать порт: рапортовать об успехе,
    # когда на localhost никого нет, — худшее, что может делать скрипт запуска.
    $ready = $false
    foreach ($attempt in 1..40) {
        Start-Sleep -Milliseconds 500
        if (Get-NetTCPConnection -LocalPort $UiPort -State Listen -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
    }
    if ($ready) { Say "             интерфейс отвечает" "DarkGray" }
    else { Say "  Фронтенд не поднялся за 20 с — посмотрите его окно" "Red" }
}

# --- бот --------------------------------------------------------------------

if (-not $NoBot) {
    $envFile = Join-Path $root ".env"
    $hasToken = (Test-Path $envFile) -and ((Get-Content $envFile -Raw) -match "TELEGRAM_BOT_TOKEN\s*=\s*\S")
    if ($hasToken) {
        # Telegram отдаёт обновления только одному потребителю getUpdates на токен.
        # Второй экземпляр не падает, а бесконечно получает Conflict и отбирает
        # сообщения у первого — со стороны выглядит как «бот отвечает через раз».
        # Учтите: один живой бот — это ДВА процесса python.exe (родитель и его
        # ребёнок), так что считать экземпляры по их числу нельзя, только по факту
        # «есть хоть один».
        $already = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*bot.bot*" })
        if ($already.Count -gt 0) {
            Say "  Бот уже запущен (pid: $($already.ProcessId -join ', ')) — второй не поднимаю" "Yellow"
            Say "  Остановить: Stop-Process -Id $($already[0].ProcessId)" "DarkGray"
        }
        else {
            Say "  Бот      -> long polling через API на :$ApiPort" "Green"
            $env:API_URL = "http://127.0.0.1:$ApiPort"
            Start-Process -FilePath $python -ArgumentList "-m", "bot.bot" -WorkingDirectory $root
        }
    }
    else {
        Say "  Бот не запущен: в .env нет TELEGRAM_BOT_TOKEN (это нормально)" "DarkGray"
    }
}

Say ""
Say "  Готово. Окна закрывать по Ctrl+C в каждом." "Cyan"
Say "  Данные лежат в data/ — те же, что видят контейнеры." "DarkGray"
Say ""
