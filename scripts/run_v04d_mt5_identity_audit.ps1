param(
    [string]$HistoryAudit = "data\mt5\calendar\v04d_history_audit.csv"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D MT5 Event Identity Audit ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    $ResolvedHistory = (Resolve-Path -LiteralPath $HistoryAudit).Path

    Write-Host '[1/3] Compile identity-audit code'
    & $Python -m compileall -q src\events\mt5_calendar_identity_audit.py tests\test_v04d_mt5_identity.py
    if ($LASTEXITCODE -ne 0) { throw 'Identity audit Python compile failed' }

    Write-Host '[2/3] Run identity-audit tests'
    & $Python -m unittest tests.test_v04d_mt5_identity -v
    if ($LASTEXITCODE -ne 0) { throw 'Identity audit tests failed' }

    Write-Host '[3/3] Audit exact MT5 event identities and compare actuals with frozen V0.4C when available'
    & $Python src\events\mt5_calendar_identity_audit.py --history-audit $ResolvedHistory
    if ($LASTEXITCODE -ne 0) { throw 'MT5 event identity audit failed' }

    Write-Host '=== V0.4D MT5 EVENT IDENTITY AUDIT COMPLETE (FAIL-CLOSED) ==='
    Write-Host 'Historical forecasts remain non-model-eligible until pre-release snapshot timing is proven.'
} finally {
    Pop-Location
}
