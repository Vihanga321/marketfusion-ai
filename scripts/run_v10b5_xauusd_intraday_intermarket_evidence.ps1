param(
    [switch]$NoPersist,
    [int]$ContextRows = 50000
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

    Write-Host 'MarketFusion V1.0B.5 XAUUSD intraday intermarket evidence research'
    Write-Host 'Prediction target: XAUUSD ONLY.'
    Write-Host 'Verified MT5 context instruments are read-only sensors, never prediction targets.'
    Write-Host 'Historical joins use exact completed-M5 broker epoch keys; no forward fill or invented DST correction.'
    Write-Host 'Previously exposed XAUUSD final tail remains quarantined.'
    Write-Host 'No model promotion or production integration is performed.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'

    $Arguments = @(
        '-m', 'src.research.v10b5_xauusd_intraday_intermarket_evidence',
        '--context-rows', $ContextRows
    )
    if ($NoPersist) { $Arguments += '--no-persist' }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B.5 XAUUSD intraday intermarket evidence research failed' }
} finally {
    Pop-Location
}
