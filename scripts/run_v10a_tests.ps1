$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host '=== MarketFusion V1.0A Tests ==='
python -m compileall -q src scripts tests
python -m unittest tests.test_v10a_audited_backfill tests.test_v05c_controlled_learning -v
python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'V1.0A validation failed' }
Write-Host 'V10A_TEST_STATUS: PASS'
