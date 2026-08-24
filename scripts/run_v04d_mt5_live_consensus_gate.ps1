param(
    [string]$SnapshotAudit = "data\mt5\calendar\v04d_snapshot_audit.csv"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D Live MT5 Consensus Gate ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    $ResolvedSnapshotAudit = (Resolve-Path -LiteralPath $SnapshotAudit).Path

    Write-Host '[1/3] Compile live consensus gate'
    & $Python -m compileall -q src\events\mt5_live_consensus_gate.py tests\test_v04d_mt5_live_consensus_gate.py
    if ($LASTEXITCODE -ne 0) { throw 'Live consensus gate compile failed' }

    Write-Host '[2/3] Run live consensus gate tests'
    & $Python -m unittest tests.test_v04d_mt5_live_consensus_gate -v
    if ($LASTEXITCODE -ne 0) { throw 'Live consensus gate tests failed' }

    Write-Host '[3/3] Verify server offset, exact target identity and strict pre-release timing'
    # Run as a package module so absolute imports such as
    # `from src.events...` resolve from the repository root on Windows.
    & $Python -m src.events.mt5_live_consensus_gate --snapshot-audit $ResolvedSnapshotAudit
    if ($LASTEXITCODE -ne 0) { throw 'Live MT5 consensus gate failed' }

    Write-Host '=== V0.4D LIVE MT5 CONSENSUS GATE COMPLETE (FAIL-CLOSED) ==='
} finally {
    Pop-Location
}
