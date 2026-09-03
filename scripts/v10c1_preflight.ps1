param(
    [switch]$RequireUninitialized
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    $Arguments = @('-m','src.runtime.v10c1_reliability','--preflight')
    if ($RequireUninitialized) { $Arguments += '--require-uninitialized' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0C.1 preflight failed' }
    Write-Host 'V10C1_PREFLIGHT_STATUS: PASS'
    Write-Host 'This check is read-only and does not record predictions or outcomes.'
} finally {
    Pop-Location
}
