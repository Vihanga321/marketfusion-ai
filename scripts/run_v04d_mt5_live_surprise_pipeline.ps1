param(
    [string]$SnapshotsTsv = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D Live MT5 Surprise Pipeline ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }

    Write-Host '[1/4] Refresh target snapshot audit + verified pre-release consensus'
    & (Join-Path $PSScriptRoot 'run_v04d_mt5_target_live_pipeline.ps1') -SnapshotsTsv $SnapshotsTsv
    if ($LASTEXITCODE -ne 0) { throw 'Target live consensus pipeline failed' }

    Write-Host '[2/4] Compile live surprise engine'
    & $Python -m compileall -q src\events\mt5_live_surprise_engine.py tests\test_v04d_mt5_live_surprise_engine.py
    if ($LASTEXITCODE -ne 0) { throw 'Live surprise compile failed' }

    Write-Host '[3/4] Run live surprise tests'
    & $Python -m unittest tests.test_v04d_mt5_live_surprise_engine -v
    if ($LASTEXITCODE -ne 0) { throw 'Live surprise tests failed' }

    Write-Host '[4/4] Build verified post-release actual + raw surprise rows'
    & $Python -m src.events.mt5_live_surprise_engine
    if ($LASTEXITCODE -ne 0) { throw 'Live surprise engine failed' }

    Write-Host '=== V0.4D LIVE MT5 SURPRISE PIPELINE COMPLETE (FAIL-CLOSED) ==='
} finally {
    Pop-Location
}
