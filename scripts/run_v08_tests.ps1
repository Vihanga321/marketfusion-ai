$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    & $Python -m compileall -q src\evaluation src\research tests\test_v08_shadow_performance.py tests\test_v08_drift.py tests\test_v08_research.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.8 compile validation failed' }
    & $Python -m unittest tests.test_v08_shadow_performance tests.test_v08_drift tests.test_v08_research -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.8 deterministic tests failed' }
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }
    Write-Host 'V08_TEST_STATUS: PASS'
} finally { Pop-Location }
