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

    Write-Host 'MarketFusion V1.0B XAUUSD controlled model research'
    Write-Host 'EURUSD artifacts are not eligible for this run.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'

    $Arguments = @('-m', 'src.research.v10b_xauusd_models')
    if ($NoPersist) { $Arguments += '--no-persist' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B XAUUSD model research failed' }
} finally {
    Pop-Location
}
