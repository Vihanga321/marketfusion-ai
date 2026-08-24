$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host '=== MarketFusion V0.9 Tests ==='
python -m compileall -q src scripts tests
python -m unittest tests.test_v09_challenger_hardening -v
python scripts\repo_safety_check.py
Write-Host 'V09_TEST_STATUS: PASS'
