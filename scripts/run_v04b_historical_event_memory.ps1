$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$ReactionData = Join-Path $Root 'data\processed\v04a_event_reactions.parquet'
$MacroData = Join-Path $Root 'data\macro\macro_safe.parquet'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4B Historical Event Memory ==='
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    if (-not (Test-Path -LiteralPath $ReactionData)) { throw "Missing frozen V0.4A reactions: $ReactionData" }
    if (-not (Test-Path -LiteralPath $MacroData)) { throw "Missing audited macro data: $MacroData" }

    Write-Host '[1/4] Compile V0.4B Python'
    & $Python -m compileall -q src\memory tests\test_v04b_historical_event_memory.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4B Python compile failed' }

    Write-Host '[2/4] Verify frozen V0.4A and build memory'
    & $Python src\memory\build_historical_event_memory.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4B memory construction failed' }

    Write-Host '[3/4] Run causal retrieval audit'
    & $Python src\memory\walk_forward_retrieval_audit.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4B walk-forward retrieval audit failed' }

    Write-Host '[4/4] Run V0.4B leakage tests'
    & $Python -m unittest tests.test_v04b_historical_event_memory -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.4B leakage tests failed' }

    Write-Host '=== V0.4B PIPELINE PASS ==='
} finally {
    Pop-Location
}
