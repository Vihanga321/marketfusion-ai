$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PidRoot = Join-Path $Root 'data\runtime\v07\pids'
$Allowed = @{
    'v05a_collector' = 'src.marketdata.v05a_runner'
    'v05b_collector' = 'src.intelligence.v05b_runner'
    'v06_runtime' = 'src.runtime.v06c_runner'
    'v07_api' = 'src.dashboard.v07_api'
    'v07_dashboard' = 'vite/bin/vite.js'
    'v08_monitor' = 'src.evaluation.v08_runner'
    'v10a_xauusd_state' = 'src.runtime.v10a_asset_state'
    'realtime_quote_EURUSD' = 'src.marketdata.realtime_quote'
    'realtime_quote_XAUUSD' = 'src.marketdata.realtime_quote'
    'v10a_xauusd_market' = 'src.marketdata.v10a_asset'
    'v10a_xauusd_shadow' = 'src.evaluation.v10a_asset_recorder'
}
if (-not (Test-Path -LiteralPath $PidRoot)) { Write-Host 'No MarketFusion V0.7 PID directory exists. Nothing to stop.'; exit 0 }
foreach ($Name in $Allowed.Keys) {
    $PidFile = Join-Path $PidRoot "$Name.json"
    if (-not (Test-Path -LiteralPath $PidFile)) { continue }
    try { $Record = Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json } catch { Write-Warning "Invalid PID record for $Name; not stopping any process."; continue }
    $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $($Record.pid)" -ErrorAction SilentlyContinue
    if ($null -eq $Info) { Remove-Item -LiteralPath $PidFile -Force; Write-Host "$Name was not running; removed stale PID record."; continue }
    if ($Info.CommandLine -notlike "*$($Allowed[$Name])*") { Write-Warning "$Name PID now belongs to another command; it was NOT stopped."; continue }
    # Virtual-environment launchers can retain a child interpreter. Stop only
    # descendants of the tracked PID whose command line carries the same
    # MarketFusion marker, deepest children first, then the recorded process.
    $Snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $Pending = @([int]$Record.pid)
    $Descendants = @()
    while ($Pending.Count -gt 0) {
        $ParentId = $Pending[0]
        if ($Pending.Count -eq 1) { $Pending = @() } else { $Pending = @($Pending[1..($Pending.Count - 1)]) }
        $Children = @($Snapshot | Where-Object { $_.ParentProcessId -eq $ParentId -and $_.CommandLine -like "*$($Allowed[$Name])*" })
        foreach ($Child in $Children) { $Descendants += $Child; $Pending += [int]$Child.ProcessId }
    }
    foreach ($Child in ($Descendants | Sort-Object ProcessId -Descending)) {
        Stop-Process -Id ([int]$Child.ProcessId) -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id ([int]$Record.pid) -Force
    Remove-Item -LiteralPath $PidFile -Force
    Write-Host "Stopped MarketFusion process $Name (PID $($Record.pid))."
}
Write-Host 'MarketFusion managed processes stopped. MT5 was not closed.'
