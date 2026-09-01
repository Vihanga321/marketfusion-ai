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

# joblib/loky may call the removed legacy WMIC command when it tries to discover
# physical CPU cores on modern Windows. Discover physical cores with PowerShell
# CIM first and pass a real worker ceiling to loky. If CIM is unavailable, use
# a conservative value strictly below the logical-core count so loky does not
# enter its WMIC physical-core probe. This changes worker-count discovery only;
# model logic, folds, targets, and promotion gates are unchanged.
$HadLokyCpuCount = Test-Path Env:\LOKY_MAX_CPU_COUNT
$PreviousLokyCpuCount = $env:LOKY_MAX_CPU_COUNT
if (-not $HadLokyCpuCount) {
    $LogicalCores = [Math]::Max(1, [Environment]::ProcessorCount)
    $PhysicalCores = 0
    try {
        $CpuSummary = Get-CimInstance Win32_Processor -ErrorAction Stop |
            Measure-Object -Property NumberOfCores -Sum
        if ($null -ne $CpuSummary.Sum) {
            $PhysicalCores = [int]$CpuSummary.Sum
        }
    } catch {
        $PhysicalCores = 0
    }

    if ($PhysicalCores -gt 0 -and $PhysicalCores -le $LogicalCores) {
        $WorkerCeiling = $PhysicalCores
        $WorkerCeilingSource = 'Windows CIM physical cores'
    } elseif ($LogicalCores -gt 1) {
        $WorkerCeiling = $LogicalCores - 1
        $WorkerCeilingSource = 'conservative logical-core fallback'
    } else {
        $WorkerCeiling = 1
        $WorkerCeilingSource = 'single-core fallback'
    }
    $env:LOKY_MAX_CPU_COUNT = [string]$WorkerCeiling
} else {
    $WorkerCeilingSource = 'existing LOKY_MAX_CPU_COUNT environment override'
}

Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }

    Write-Host 'MarketFusion V1.0B XAUUSD controlled model research'
    Write-Host 'EURUSD artifacts are not eligible for this run.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'
    Write-Host "Research CPU worker ceiling: $env:LOKY_MAX_CPU_COUNT ($WorkerCeilingSource)"

    $Arguments = @('-m', 'src.research.v10b_xauusd_models')
    if ($NoPersist) { $Arguments += '--no-persist' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B XAUUSD model research failed' }
} finally {
    Pop-Location
    if ($HadLokyCpuCount) { $env:LOKY_MAX_CPU_COUNT = $PreviousLokyCpuCount }
    else { Remove-Item Env:\LOKY_MAX_CPU_COUNT -ErrorAction SilentlyContinue }
}
