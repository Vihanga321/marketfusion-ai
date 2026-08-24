$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Set-Location $Root

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
& $Python -m src.inference.v06a_runner --status
