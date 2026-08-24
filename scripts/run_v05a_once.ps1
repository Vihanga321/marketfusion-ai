$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5A One-Cycle Local MT5 Test ==='
    Write-Host 'Requirement: MetaTrader 5 desktop must be open and logged in.'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    & $Python -m src.marketdata.v05a_runner --once
    if ($LASTEXITCODE -ne 0) { throw 'V0.5A one-cycle collection failed' }
    Write-Host '=== V0.5A ONE-CYCLE COMPLETE ==='
} finally {
    Pop-Location
}
