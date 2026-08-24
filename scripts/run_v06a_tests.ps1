$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

Write-Host '=== MarketFusion V0.6A Tests ==='
python -m unittest tests.test_v06a_shadow_inference -v
if ($LASTEXITCODE -ne 0) { throw 'V0.6A focused tests failed' }
python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety failed' }
Write-Host 'V06A_TEST_STATUS: PASS'
