param([int]$IntervalSeconds = 15)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    & $Python -m src.runtime.v06c_runner --continuous --interval-seconds $IntervalSeconds
    if ($LASTEXITCODE -ne 0) { throw 'V0.6 unified runtime stopped with an error' }
} finally { Pop-Location }
