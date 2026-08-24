param(
    [string]$HistoryTsv,
    [string]$SnapshotsTsv
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D Free MT5 Economic Calendar Audit ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    if ([string]::IsNullOrWhiteSpace($HistoryTsv) -and [string]::IsNullOrWhiteSpace($SnapshotsTsv)) {
        throw 'Provide -HistoryTsv and/or -SnapshotsTsv.'
    }

    Write-Host '[1/3] Compile V0.4D Python'
    & $Python -m compileall -q src\events\mt5_calendar_importer.py tests\test_v04d_mt5_calendar.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D Python compile failed' }

    Write-Host '[2/3] Run fail-closed V0.4D tests'
    & $Python -m unittest tests.test_v04d_mt5_calendar -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D unit tests failed' }

    $RunArgs = @()
    if (-not [string]::IsNullOrWhiteSpace($HistoryTsv)) {
        $ResolvedHistory = (Resolve-Path -LiteralPath $HistoryTsv).Path
        $RunArgs += @('--history', $ResolvedHistory)
    }
    if (-not [string]::IsNullOrWhiteSpace($SnapshotsTsv)) {
        $ResolvedSnapshots = (Resolve-Path -LiteralPath $SnapshotsTsv).Path
        $RunArgs += @('--snapshots', $ResolvedSnapshots)
    }

    Write-Host '[3/3] Import and audit MT5 calendar data'
    & $Python src\events\mt5_calendar_importer.py @RunArgs
    if ($LASTEXITCODE -ne 0) { throw 'V0.4D MT5 calendar audit failed' }

    Write-Host '=== V0.4D FREE MT5 CALENDAR FOUNDATION PASS (FAIL-CLOSED) ==='
    Write-Host 'Historical MT5 forecast values are NOT yet model-eligible consensus.'
    Write-Host 'Future snapshots have a verified capture time, but event identity still requires an audited mapping.'
} finally {
    Pop-Location
}
