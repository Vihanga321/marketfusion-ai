param(
    [switch]$NoPersist
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtual-environment Python: $Python"
}

Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }

    Write-Host 'MarketFusion V1.0B.4 XAUUSD intraday intermarket availability audit'
    Write-Host 'Prediction target: XAUUSD ONLY.'
    Write-Host 'Other MT5 symbols are read-only context candidates, never trading targets.'
    Write-Host 'Audit probes broker symbol identity, live timestamp evidence, and bounded completed M5/M15 history.'
    Write-Host 'No model training or promotion is performed.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'

    $Arguments = @('-m', 'src.research.v10b4_xauusd_intraday_context_audit')
    if ($NoPersist) { $Arguments += '--no-persist' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B.4 XAUUSD intraday context audit failed' }
} finally {
    Pop-Location
}
