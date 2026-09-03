$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
Set-Location $repo
Write-Host '=== MarketFusion V0.9A.1 Tests ==='
& $python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { throw 'V0.9A.1 compilation failed' }
& $python -m unittest tests.test_v09a_engines tests.test_v09a1_market_intelligence -v
if ($LASTEXITCODE -ne 0) { throw 'V0.9A.1 deterministic tests failed' }
& $python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
Write-Host 'V09A1_TEST_STATUS: PASS'
