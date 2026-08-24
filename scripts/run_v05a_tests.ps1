$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5A Test Gate ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }

    Write-Host '[1/3] Compile V0.5A Python'
    & $Python -m compileall -q src\marketdata tests\test_v05a_continuous_market_data.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.5A compile failed' }

    Write-Host '[2/3] Run V0.5A unit tests'
    & $Python -m unittest tests.test_v05a_continuous_market_data -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.5A unit tests failed' }

    Write-Host '[3/3] Repository read-only/safety preflight'
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety preflight failed' }

    Write-Host 'V05A_TEST_STATUS: PASS'
} finally {
    Pop-Location
}
