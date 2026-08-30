$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
Set-Location $repo
Write-Host '=== MarketFusion V0.9A.1 Tests ==='
& $python -m compileall -q src scripts tests
& $python -m unittest tests.test_v09a_engines tests.test_v09a1_market_intelligence -v
& $python scripts\repo_safety_check.py
Write-Host 'V09A1_TEST_STATUS: PASS'
