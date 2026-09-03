$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing virtual-environment Python: $python" }
Set-Location $repo
Write-Host '=== MarketFusion V1.0A Tests ==='
& $python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { throw 'V1.0A compilation failed' }
& $python -m unittest tests.test_v10a_audited_backfill tests.test_v05c_controlled_learning -v
if ($LASTEXITCODE -ne 0) { throw 'V1.0A deterministic tests failed' }
& $python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
Write-Host 'V10A_TEST_STATUS: PASS'
