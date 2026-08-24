$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5C Controlled Daily Learning ==='
    Write-Host 'Research predictions only. Trading execution is disabled.'
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    Write-Host '[1/3] Repository safety'
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety failed' }
    Write-Host '[2/3] V0.5A deterministic integrity gate'
    & (Join-Path $PSScriptRoot 'run_v05a_tests.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'V0.5A integrity gate failed' }
    Write-Host '[3/3] Matured causal training, walk-forward evaluation, and immutable registry update'
    & $Python -m src.learning.v05c_runner
    if ($LASTEXITCODE -ne 0) { throw 'V0.5C daily learning failed closed' }
    Write-Host '=== V0.5C DAILY LEARNING COMPLETE ==='
} finally {
    Pop-Location
}
