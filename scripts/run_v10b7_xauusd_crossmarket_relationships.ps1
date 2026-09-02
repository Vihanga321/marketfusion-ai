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

    Write-Host 'MarketFusion V1.0B.7 XAUUSD causal cross-market relationship research'
    Write-Host 'Prediction target: XAUUSD ONLY.'
    Write-Host 'Frozen relationship contracts only: 15m Silver and 30m FX/USD.'
    Write-Host 'Enhanced relationship features are compared against the same raw-context model on identical rows.'
    Write-Host 'This is the last historical exploratory phase on the inspected sample.'
    Write-Host 'The final XAUUSD tail remains quarantined; this phase is promotion-ineligible.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'

    $Arguments = @(
        '-m', 'src.research.v10b7_xauusd_crossmarket_relationships',
        '--context-rows', $ContextRows
    )
    if ($NoPersist) { $Arguments += '--no-persist' }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'V1.0B.7 XAUUSD cross-market relationship research failed' }
} finally {
    Pop-Location
}
