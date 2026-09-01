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

    Write-Host 'MarketFusion V1.0B.1 XAUUSD selective target research'
    Write-Host 'This phase compares WAIT-zone and horizon contracts only.'
    Write-Host 'No model promotion is performed.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'
    Write-Host "Research CPU worker ceiling: $env:LOKY_MAX_CPU_COUNT ($WorkerCeilingSource)"

    $Arguments = @('-m', 'src.research.v10b1_xauusd_targets')
    if ($NoPersist) { $Arguments += '--no-persist' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B.1 XAUUSD target research failed' }
} finally {
    Pop-Location
    if ($HadLokyCpuCount) { $env:LOKY_MAX_CPU_COUNT = $PreviousLokyCpuCount }
    else { Remove-Item Env:\LOKY_MAX_CPU_COUNT -ErrorAction SilentlyContinue }
}
