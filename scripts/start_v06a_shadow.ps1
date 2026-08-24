param(
    [int]$IntervalSeconds = 30
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

Write-Host "=== Starting MarketFusion V0.6A every $IntervalSeconds seconds ==="
Write-Host 'Shadow inference only. Trading execution is disabled.'
python -m src.inference.v06a_runner --continuous --interval-seconds $IntervalSeconds
