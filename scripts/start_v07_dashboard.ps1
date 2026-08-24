param([int]$Port = 5173)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Dashboard = Join-Path $Root 'dashboard'
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
    npm run dev -- --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'V0.7 dashboard stopped with an error' }
} finally { Pop-Location }
