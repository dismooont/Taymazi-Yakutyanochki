<#
.SYNOPSIS
    Скачивает и распаковывает MS COCO val2017 (изображения + аннотации) под Windows.
    PowerShell-аналог scripts/download_coco_val2017.sh.

.DESCRIPTION
    Качает val2017.zip (~1 ГБ) и annotations_trainval2017.zip (~240 МБ),
    распаковывает в папку назначения. По умолчанию — .\data.

    Итоговая структура:
      <dest>\images\val2017\*.jpg
      <dest>\annotations\captions_val2017.json

.PARAMETER DestDir
    Папка назначения. По умолчанию "data".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_coco_val2017.ps1
    powershell -ExecutionPolicy Bypass -File scripts\download_coco_val2017.ps1 -DestDir data
#>

param(
    [string]$DestDir = "data"
)

$ErrorActionPreference = "Stop"

$imagesUrl = "http://images.cocodataset.org/zips/val2017.zip"
$annUrl     = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

$imagesDir = Join-Path $DestDir "images"
New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null

$tmpImages = Join-Path $env:TEMP "val2017.zip"
$tmpAnn     = Join-Path $env:TEMP "annotations_trainval2017.zip"

# curl.exe (встроен в Windows 10/11) заметно быстрее и надёжнее на больших файлах,
# чем Invoke-WebRequest. Если его нет — откатываемся на Invoke-WebRequest.
function Get-File($url, $outFile) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        # -C - докачивает с места обрыва (важно: прокси рвёт соединение на больших файлах),
        # --retry/--retry-all-errors повторяют попытку при обрыве (curl error 18 и пр.).
        & curl.exe -L --fail -C - --retry 20 --retry-delay 5 --retry-all-errors -o $outFile $url
        if ($LASTEXITCODE -ne 0) { throw "curl завершился с кодом $LASTEXITCODE при скачивании $url" }
    } else {
        $ProgressPreference = "SilentlyContinue"  # без прогресс-бара I-WR работает в разы быстрее
        Invoke-WebRequest -Uri $url -OutFile $outFile
    }
}

Write-Host "Скачивание изображений val2017 (~1 ГБ)..."
Get-File $imagesUrl $tmpImages

Write-Host "Скачивание аннотаций (~240 МБ)..."
Get-File $annUrl $tmpAnn

Write-Host "Распаковка изображений..."
Expand-Archive -Path $tmpImages -DestinationPath $imagesDir -Force

Write-Host "Распаковка аннотаций..."
Expand-Archive -Path $tmpAnn -DestinationPath $DestDir -Force

Remove-Item -Force $tmpImages, $tmpAnn -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Готово. Структура:"
Write-Host "  $DestDir\images\val2017\*.jpg"
Write-Host "  $DestDir\annotations\captions_val2017.json"
Write-Host ""
Write-Host "Дальше сконвертируйте аннотации в CSV:"
Write-Host "  .\.venv\Scripts\python.exe scripts\convert_coco_captions.py ``"
Write-Host "      --input $DestDir\annotations\captions_val2017.json ``"
Write-Host "      --output $DestDir\captions.csv"
