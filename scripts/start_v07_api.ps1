param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    Write-Host "Starting MarketFusion V0.7 API at http://127.0.0.1:$Port"
    Write-Host 'Localhost-only, read-only, and no trading endpoints.'
    & $Python -m src.dashboard.v07_api --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'V0.7 API stopped with an error' }
} finally { Pop-Location }
