$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Set-Location $Root

Write-Host '=== MarketFusion V0.6A One Shadow Cycle ==='
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
& $Python -m src.inference.v06a_runner
if ($LASTEXITCODE -ne 0) { throw 'V0.6A one-cycle runner failed' }
