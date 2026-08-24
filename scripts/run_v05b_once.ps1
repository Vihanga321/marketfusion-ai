$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
Write-Host '=== MarketFusion V0.5B One Cycle ==='
python -m src.intelligence.v05b_runner --once
if ($LASTEXITCODE -ne 0) { throw 'V0.5B one-cycle run failed' }
Write-Host '=== V0.5B ONE-CYCLE COMPLETE ==='
