param(
    [ValidateSet('EURUSD','XAUUSD')][string]$Symbol = 'XAUUSD',
    [switch]$SkipRuntime,
    [switch]$NoBrowser,
    [ValidateRange(1, 65535)][int]$DashboardPort = 4173
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PidRoot = Join-Path $Root 'data\runtime\v07\pids'
$RootPattern = [regex]::Escape($Root)

function Stop-OwnedListener {
    param(
        [Parameter(Mandatory=$true)][int]$Port,
        [Parameter(Mandatory=$true)][string]$AllowedPattern,
        [Parameter(Mandatory=$true)][string]$ExpectedPidFile,
        [switch]$ValidateOnly
    )

    $Listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($Listener in $Listeners) {
        $OwnerProcessId = [int]$Listener.OwningProcess
        $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $OwnerProcessId" -ErrorAction SilentlyContinue
        if ($null -eq $Info) { continue }
        $CommandLine = [string]$Info.CommandLine
        $PidRecord = $null
        if (Test-Path -LiteralPath $ExpectedPidFile) {
            try { $PidRecord = Get-Content -Raw -LiteralPath $ExpectedPidFile | ConvertFrom-Json } catch { $PidRecord = $null }
        }
        $RecordedProcessId = if ($null -ne $PidRecord -and $null -ne $PidRecord.pid) { [int]$PidRecord.pid } else { 0 }
        $RecordedTreeIds = @()
        if ($RecordedProcessId -gt 0) {
            $Snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
            $Pending = @($RecordedProcessId)
            while ($Pending.Count -gt 0) {
                $CandidateId = [int]$Pending[0]
                if ($Pending.Count -eq 1) { $Pending = @() } else { $Pending = @($Pending[1..($Pending.Count - 1)]) }
                if ($RecordedTreeIds -contains $CandidateId) { continue }
                $RecordedTreeIds += $CandidateId
                $Pending += @($Snapshot | Where-Object { $_.ParentProcessId -eq $CandidateId } | ForEach-Object { [int]$_.ProcessId })
            }
        }
        $OwnedByRecord = $RecordedTreeIds -contains $OwnerProcessId
        $OwnedByRepository = $CommandLine -match $RootPattern
        if ($CommandLine -notmatch $AllowedPattern -or (-not $OwnedByRecord -and -not $OwnedByRepository)) {
            throw "TCP port $Port is already owned by non-MarketFusion PID $OwnerProcessId. Command: $CommandLine"
        }
        if ($ValidateOnly) { continue }
        Write-Host "Stopping stale MarketFusion listener on port $Port (PID $OwnerProcessId)."
        Stop-Process -Id $OwnerProcessId -Force -ErrorAction Stop
    }
}

function Stop-OwnedProcessRecord {
    param(
        [Parameter(Mandatory=$true)][string]$PidFile,
        [Parameter(Mandatory=$true)][string]$AllowedPattern,
        [Parameter(Mandatory=$true)][string]$Name
    )
    if (-not (Test-Path -LiteralPath $PidFile)) { return }
    $Record = $null
    try { $Record = Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json } catch { $Record = $null }
    if ($null -eq $Record -or $null -eq $Record.pid) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return
    }
    $ProcessId = [int]$Record.pid
    $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $Info) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return
    }
    $CommandLine = [string]$Info.CommandLine
    if ($CommandLine -notmatch $AllowedPattern -or $CommandLine -notmatch $RootPattern) {
        throw "$Name PID record points to a process that is not owned by this MarketFusion repository. PID $ProcessId. Command: $CommandLine"
    }
    Write-Host "Restarting $Name (PID $ProcessId)."
    Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

Push-Location $Root
try {
    # Validate every required port before stopping anything, avoiding a partial
    # restart if another application owns either listener.
    Stop-OwnedListener -Port 8765 -AllowedPattern 'src\.dashboard\.(full_api|v07_api)' -ExpectedPidFile (Join-Path $PidRoot 'v07_api.json') -ValidateOnly
    Stop-OwnedListener -Port $DashboardPort -AllowedPattern '(vite.*dashboard|dashboard.*vite)' -ExpectedPidFile (Join-Path $PidRoot 'v07_dashboard.json') -ValidateOnly

    if (-not $SkipRuntime) {
        $QuotePidFile = Join-Path $PidRoot "realtime_quote_$($Symbol.ToLowerInvariant()).json"
        Stop-OwnedProcessRecord -PidFile $QuotePidFile -AllowedPattern 'src\.marketdata\.realtime_quote' -Name "realtime quote $Symbol"
        if ($Symbol -eq 'XAUUSD') {
            Stop-OwnedProcessRecord -PidFile (Join-Path $PidRoot 'v10a_xauusd_state.json') -AllowedPattern 'src\.runtime\.v10a_asset_state' -Name 'XAUUSD state'
        }
    }

    Stop-OwnedListener -Port 8765 -AllowedPattern 'src\.dashboard\.(full_api|v07_api)' -ExpectedPidFile (Join-Path $PidRoot 'v07_api.json')
    Stop-OwnedListener -Port $DashboardPort -AllowedPattern '(vite.*dashboard|dashboard.*vite)' -ExpectedPidFile (Join-Path $PidRoot 'v07_dashboard.json')

    Remove-Item (Join-Path $PidRoot 'v07_api.json') -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $PidRoot 'v07_dashboard.json') -Force -ErrorAction SilentlyContinue

    $Launcher = Join-Path $Root 'scripts\start_marketfusion_dashboard.ps1'
    $LaunchParams = @{
        Symbol = $Symbol
        DashboardPort = $DashboardPort
        SkipRuntime = $SkipRuntime
        NoBrowser = $NoBrowser
    }
    & $Launcher @LaunchParams
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
