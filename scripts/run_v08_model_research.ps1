$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    Write-Host 'Running manual V0.8 research experiments. This does not modify the live registry.'
    & $Python -m src.research.v08_experiments
    if ($LASTEXITCODE -ne 0) { throw 'V0.8 model research failed' }
} finally { Pop-Location }
