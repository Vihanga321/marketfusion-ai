param([int]$IntervalSeconds = 300)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    Write-Host "Starting read-only V0.8 forward monitor every $IntervalSeconds seconds."
    & $Python -m src.evaluation.v08_runner --continuous --interval-seconds $IntervalSeconds
    if ($LASTEXITCODE -ne 0) { throw 'V0.8 monitor stopped with an error' }
} finally { Pop-Location }
