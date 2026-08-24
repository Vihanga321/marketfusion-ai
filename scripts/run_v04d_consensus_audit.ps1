param(
    [switch]$InitializeFingerprint,
    [switch]$FullCollection
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$Memory = Join-Path $Root 'data\processed\v04c_historical_event_memory.parquet'
$Values = Join-Path $Root 'data\processed\v04c_event_values.parquet'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D Historical Consensus Intelligence ==='
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    if (-not (Test-Path -LiteralPath $Memory)) { throw "Missing frozen V0.4C memory: $Memory" }
    if (-not (Test-Path -LiteralPath $Values)) { throw "Missing frozen V0.4C event values: $Values" }

    Write-Host '[1/4] Compile V0.4D Python'
    & $Python -m compileall -q src\events tests\test_v04d_consensus.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D Python compile failed' }

    Write-Host '[2/4] Run fail-closed consensus unit tests'
    & $Python -m unittest tests.test_v04d_consensus -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D unit tests failed' }

    Write-Host '[3/4] Run provider audit pipeline'
    $Arguments = @('src\events\run_v04d_consensus_audit.py')
    if ($InitializeFingerprint) { $Arguments += '--initialize-fingerprint' }
    if ($FullCollection) { $Arguments += '--full-collection' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D provider audit failed' }

    Write-Host '[4/4] Credential presence (boolean only)'
    Write-Host "TRADING_ECONOMICS_CREDENTIAL_PRESENT: $([bool]$env:TRADING_ECONOMICS_API_KEY)"
    Write-Host '=== V0.4D PIPELINE COMPLETE ==='
} finally {
    Pop-Location
}
