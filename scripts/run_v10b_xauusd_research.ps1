param(
    [switch]$NoPersist
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtual-environment Python: $Python"
}

# joblib/loky tries the legacy WMIC command on some modern Windows systems
# when discovering physical CPU cores. WMIC is optional/removed on current
# Windows builds, so provide the already-known logical processor count instead.
# This changes only worker-count discovery; it does not change model logic.
$HadLokyCpuCount = Test-Path Env:\LOKY_MAX_CPU_COUNT
$PreviousLokyCpuCount = $env:LOKY_MAX_CPU_COUNT
if (-not $HadLokyCpuCount) {
    $env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount
}

Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }

    Write-Host 'MarketFusion V1.0B XAUUSD controlled model research'
    Write-Host 'EURUSD artifacts are not eligible for this run.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'
    Write-Host "Research CPU worker ceiling: $env:LOKY_MAX_CPU_COUNT logical processors"

    $Arguments = @('-m', 'src.research.v10b_xauusd_models')
    if ($NoPersist) { $Arguments += '--no-persist' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B XAUUSD model research failed' }
} finally {
    Pop-Location
    if ($HadLokyCpuCount) { $env:LOKY_MAX_CPU_COUNT = $PreviousLokyCpuCount }
    else { Remove-Item Env:\LOKY_MAX_CPU_COUNT -ErrorAction SilentlyContinue }
}
