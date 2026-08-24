param(
    [int]$IntervalSeconds = 300
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root
Write-Host "=== Starting MarketFusion V0.5B every $IntervalSeconds seconds ==="
Write-Host 'This collector is read-only and does not trade.'
python -m src.intelligence.v05b_runner --interval $IntervalSeconds
if ($LASTEXITCODE -ne 0) { throw 'V0.5B continuous runner stopped with an error' }
