$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Set-Location $Root

Write-Host '=== MarketFusion V0.6A Tests ==='
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
& $Python -m unittest tests.test_v06a_shadow_inference -v
if ($LASTEXITCODE -ne 0) { throw 'V0.6A focused tests failed' }
& $Python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety failed' }
Write-Host 'V06A_TEST_STATUS: PASS'
