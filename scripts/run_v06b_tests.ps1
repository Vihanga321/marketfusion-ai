$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    Write-Host '=== MarketFusion V0.6B/V0.6C Focused Tests ==='
    & $Python -m unittest tests.test_v06b_fusion tests.test_v06c_runtime -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.6B/V0.6C focused tests failed' }
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety failed' }
    Write-Host 'V06B_TEST_STATUS: PASS'
} finally {
    Pop-Location
}
