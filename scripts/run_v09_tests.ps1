$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
Set-Location $repo
Write-Host '=== MarketFusion V0.9 Tests ==='
& $python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { throw 'V0.9 compilation failed' }
& $python -m unittest tests.test_v09_challenger_hardening tests.test_v09_reproduction -v
if ($LASTEXITCODE -ne 0) { throw 'V0.9 deterministic tests failed' }
& $python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
Write-Host 'V09_TEST_STATUS: PASS'
