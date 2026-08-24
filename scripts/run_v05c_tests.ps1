$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5C Deterministic Tests ==='
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    & $Python -m compileall -q src\learning tests\test_v05c_controlled_learning.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.5C compile failed' }
    & $Python -m unittest tests.test_v05c_controlled_learning -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.5C tests failed' }
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
    Write-Host 'V05C_TEST_STATUS: PASS'
} finally {
    Pop-Location
}
