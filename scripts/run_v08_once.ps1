$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    & $Python -m src.evaluation.v08_runner
    if ($LASTEXITCODE -ne 0) { throw 'V0.8 monitor cycle failed' }
} finally { Pop-Location }
