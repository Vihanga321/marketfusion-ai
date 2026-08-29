param(
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [ValidateRange(1, 65535)][int]$DashboardPort = 4173
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$HadDashboardPortEnvironment = Test-Path Env:\MARKETFUSION_DASHBOARD_PORT
$PreviousDashboardPortEnvironment = $env:MARKETFUSION_DASHBOARD_PORT
$env:MARKETFUSION_DASHBOARD_PORT = [string]$DashboardPort
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    Write-Host "Starting MarketFusion V0.7 API at http://127.0.0.1:$Port"
    Write-Host 'Localhost-only, read-only, and no trading endpoints.'
    & $Python -m src.dashboard.v07_api --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'V0.7 API stopped with an error' }
} finally {
    if ($HadDashboardPortEnvironment) { $env:MARKETFUSION_DASHBOARD_PORT = $PreviousDashboardPortEnvironment }
    else { Remove-Item Env:\MARKETFUSION_DASHBOARD_PORT -ErrorAction SilentlyContinue }
    Pop-Location
}
