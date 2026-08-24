$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

Write-Host '=== MarketFusion V0.6A One Shadow Cycle ==='
python -m src.inference.v06a_runner
if ($LASTEXITCODE -ne 0) { throw 'V0.6A one-cycle runner failed' }
