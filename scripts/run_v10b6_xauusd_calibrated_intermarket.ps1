param(
    [switch]$NoPersist,
    [int]$ContextRows = 50000
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtual-environment Python: $Python"
}

Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }

    try {
        $PhysicalCores = @(
            Get-CimInstance Win32_Processor -ErrorAction Stop |
            ForEach-Object { [int]$_.NumberOfCores }
        ) | Measure-Object -Sum
        if ($PhysicalCores.Sum -and [int]$PhysicalCores.Sum -gt 0) {
            $env:LOKY_MAX_CPU_COUNT = [string][int]$PhysicalCores.Sum
            Write-Host "Research CPU worker ceiling: $($env:LOKY_MAX_CPU_COUNT) (Windows CIM physical cores)"
        }
    } catch {
        Write-Host 'Research CPU worker ceiling: joblib default (CIM physical-core query unavailable)'
    }

    Write-Host 'MarketFusion V1.0B.6 XAUUSD calibrated intraday intermarket research'
    Write-Host 'Prediction target: XAUUSD ONLY.'
    Write-Host 'Near-pass V1.0B.5 contracts only; no new model families or threshold lowering.'
    Write-Host 'Calibration uses an inner trailing chronological slice inside each outer training fold.'
    Write-Host 'Base-training targets are purged before calibration; final XAUUSD tail remains quarantined.'
    Write-Host 'This phase is exploratory and promotion-ineligible even if evidence is found.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'

    $Arguments = @(
        '-m', 'src.research.v10b6_xauusd_calibrated_intermarket',
        '--context-rows', $ContextRows
    )
    if ($NoPersist) { $Arguments += '--no-persist' }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B.6 XAUUSD calibrated intermarket research failed' }
} finally {
    Pop-Location
}
