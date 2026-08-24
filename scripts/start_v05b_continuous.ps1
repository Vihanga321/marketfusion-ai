param(
    [int]$IntervalSeconds = 300
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host "=== Starting MarketFusion V0.5B every $IntervalSeconds seconds ==="
    Write-Host 'This collector is read-only and does not trade.'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    & $Python -m src.intelligence.v05b_runner --interval $IntervalSeconds
    if ($LASTEXITCODE -ne 0) { throw 'V0.5B continuous runner stopped with an error' }
} finally {
    Pop-Location
}
