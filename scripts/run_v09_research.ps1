$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host '=== MarketFusion V0.9 Challenger Hardening Research ==='
Write-Host 'Research only. Formal promotion remains V0.5C. Trading execution is disabled.'
python -m src.research.v09_entry
Write-Host '=== V0.9 RESEARCH COMPLETE ==='
