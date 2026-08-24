$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    & $Python -m src.fusion.v06b_runner
    if ($LASTEXITCODE -ne 0) { throw 'V0.6B one-cycle runner failed' }
} finally { Pop-Location }
