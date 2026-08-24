param(
    [int]$PollSeconds = 15,
    [int]$OverlapBars = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5A Continuous EURUSD Collector ==='
    Write-Host 'MetaTrader 5 must stay open and logged in.'
    Write-Host 'Read-only: this process cannot place or modify trades.'
    Write-Host "Poll interval: $PollSeconds seconds"
    Write-Host 'Press Ctrl+C to stop safely.'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    & $Python -m src.marketdata.v05a_runner --poll-seconds $PollSeconds --overlap-bars $OverlapBars
    if ($LASTEXITCODE -ne 0) { throw 'V0.5A continuous collector stopped with an error' }
} finally {
    Pop-Location
}
