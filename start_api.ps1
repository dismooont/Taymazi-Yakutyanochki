Set-Location $PSScriptRoot
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:NO_PROXY = "*"
$env:PYTHONUTF8 = "1"
& ".\.venv\Scripts\python.exe" -m uvicorn web.app:app --port 8000
