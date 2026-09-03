param(
    [ValidateRange(30, 3600)][int]$IntervalSeconds = 60,
    [ValidateRange(30, 3600)][int]$MaxRetrySeconds = 300,
    [switch]$Once,
    [switch]$Status
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }

    if ($Status) {
        & $Python -m src.runtime.v10c1_reliability --status
        exit $LASTEXITCODE
    }

    Write-Host 'MarketFusion V1.0C.1 reliable forward collector'
    Write-Host 'Automatic execution: DISABLED'
    Write-Host 'Runtime: SHADOW_ADVISORY_ONLY'
    Write-Host 'Manual confirmation: REQUIRED'
    Write-Host 'Backfill: PROHIBITED'
    Write-Host 'Initialization is NOT performed by this launcher.'

    $Arguments = @('-m','src.runtime.v10c1_reliability','--interval-seconds',[string]$IntervalSeconds,'--max-retry-seconds',[string]$MaxRetrySeconds)
    if (-not $Once) { $Arguments += '--continuous' }

    & $Python @Arguments
    if ($LASTEXITCODE -notin @(0,130)) {
        throw "V1.0C.1 reliable collector exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
