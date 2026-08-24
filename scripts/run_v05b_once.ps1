$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.5B One Cycle ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    & $Python -m src.intelligence.v05b_runner --once
    if ($LASTEXITCODE -ne 0) { throw 'V0.5B one-cycle run failed' }
    Write-Host '=== V0.5B ONE-CYCLE COMPLETE ==='
} finally {
    Pop-Location
}
