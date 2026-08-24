$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

Write-Host '=== MarketFusion V0.5B Tests ==='
python -m compileall -q src\intelligence tests\test_v05b_continuous_intelligence.py
if ($LASTEXITCODE -ne 0) { throw 'V0.5B compile failed' }
python -m unittest tests.test_v05b_continuous_intelligence -v
if ($LASTEXITCODE -ne 0) { throw 'V0.5B tests failed' }
python .\scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }
Write-Host 'V05B_TEST_STATUS: PASS'
