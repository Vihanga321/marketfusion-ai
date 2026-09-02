param(
    [switch]$Initialize,
    [switch]$Status,
    [switch]$Continuous,
    [int]$IntervalSeconds = 60,
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
            Write-Host "Forward-validation CPU worker ceiling: $($env:LOKY_MAX_CPU_COUNT) (Windows CIM physical cores)"
        }
    } catch {
        Write-Host 'Forward-validation CPU worker ceiling: joblib default (CIM query unavailable)'
    }

    Write-Host 'MarketFusion V1.0C XAUUSD frozen forward point-in-time validation'
    Write-Host 'Prediction target: XAUUSD ONLY.'
    Write-Host 'Frozen hypotheses: 15m Silver LR and 30m FX/USD HGB, each with paired price-only baseline.'
    Write-Host 'Historical tuning is STOPPED. No retraining, recalibration, feature search, or threshold changes during the forward window.'
    Write-Host 'Minimum evaluation sample is frozen before collection: 20 active days, 500 matured observations, 400 directional outcomes per contract.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY; manual confirmation remains REQUIRED.'

    $Arguments = @('-m', 'src.evaluation.v10c_xauusd_forward_validation')
    if ($Initialize) {
        $Arguments += '--initialize'
        $Arguments += @('--context-rows', $ContextRows)
    } elseif ($Status) {
        $Arguments += '--status'
    } elseif ($Continuous) {
        $Arguments += '--continuous'
        $Arguments += @('--interval-seconds', $IntervalSeconds)
    }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0C XAUUSD forward validation failed' }
} finally {
    Pop-Location
}
