param([ValidateRange(1, 65535)][int]$Port = 4173)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Dashboard = Join-Path $Root 'dashboard'
$HadDashboardPortEnvironment = Test-Path Env:\MARKETFUSION_DASHBOARD_PORT
$PreviousDashboardPortEnvironment = $env:MARKETFUSION_DASHBOARD_PORT
$env:MARKETFUSION_DASHBOARD_PORT = [string]$Port
Push-Location $Dashboard
try {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js is not installed or not on PATH' }
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'npm is not installed or not on PATH' }
    if (-not (Test-Path -LiteralPath (Join-Path $Dashboard 'node_modules'))) {
        Write-Host 'Installing dashboard dependencies from package-lock.json...'
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'Dashboard npm install failed' }
    }
    Write-Host "Starting MarketFusion dashboard at http://127.0.0.1:$Port"
    npm run dev -- --host 127.0.0.1 --port $Port --strictPort
    if ($LASTEXITCODE -ne 0) { throw 'V0.7 dashboard stopped with an error' }
} finally {
    if ($HadDashboardPortEnvironment) { $env:MARKETFUSION_DASHBOARD_PORT = $PreviousDashboardPortEnvironment }
    else { Remove-Item Env:\MARKETFUSION_DASHBOARD_PORT -ErrorAction SilentlyContinue }
    Pop-Location
}
