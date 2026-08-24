param(
    [int]$IntervalSeconds = 30
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Set-Location $Root

Write-Host "=== Starting MarketFusion V0.6A every $IntervalSeconds seconds ==="
Write-Host 'Shadow inference only. Trading execution is disabled.'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
& $Python -m src.inference.v06a_runner --continuous --interval-seconds $IntervalSeconds
