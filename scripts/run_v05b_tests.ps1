$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5B Tests ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    & $Python -m compileall -q src\intelligence tests\test_v05b_continuous_intelligence.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.5B compile failed' }
    & $Python -m unittest tests.test_v05b_continuous_intelligence -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.5B tests failed' }
    & $Python .\scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
    Write-Host 'V05B_TEST_STATUS: PASS'
} finally {
    Pop-Location
}
