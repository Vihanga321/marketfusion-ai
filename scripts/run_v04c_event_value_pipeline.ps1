$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$V04BMemory = Join-Path $Root 'data\processed\v04b_historical_event_memory.parquet'
$AlfredData = Join-Path $Root 'data\macro\fred_macro.parquet'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4C Event Value and Surprise Intelligence ==='
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    if (-not (Test-Path -LiteralPath $V04BMemory)) { throw "Missing frozen V0.4B memory: $V04BMemory" }
    if (-not (Test-Path -LiteralPath $AlfredData)) { throw "Missing audited local ALFRED data: $AlfredData" }

    Write-Host '[1/5] Compile V0.4C Python'
    & $Python -m compileall -q src\events tests\test_v04c_event_values.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4C Python compile failed' }

    Write-Host '[2/5] Verify frozen V0.4B and build normalized values'
    & $Python src\events\build_v04c_event_values.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4C event-value construction failed' }

    Write-Host '[3/5] Build decision-mode features and enriched memory'
    & $Python src\events\build_v04c_enriched_memory.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4C enriched-memory construction failed' }

    Write-Host '[4/5] Run surprise walk-forward sample gate'
    & $Python src\events\surprise_walk_forward_audit.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4C surprise audit failed' }

    Write-Host '[5/5] Run V0.4C chronology/leakage tests'
    & $Python -m unittest tests.test_v04c_event_values -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.4C unit tests failed' }

    Write-Host '=== V0.4C PIPELINE PASS (SURPRISE SAMPLE INSUFFICIENT) ==='
} finally {
    Pop-Location
}
