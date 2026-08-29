$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host '=== MarketFusion V1.0A READ-ONLY Historical Backfill Audit ==='
Write-Host 'MT5 must be open. No local market store will be modified by this command.'
python -m src.backfill.v10a_runner
if ($LASTEXITCODE -ne 0) { throw 'V1.0A audit failed' }
Write-Host '=== V1.0A AUDIT COMPLETE ==='
