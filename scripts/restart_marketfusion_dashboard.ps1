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

function Stop-OwnedListener {
    param(
        [Parameter(Mandatory=$true)][int]$Port,
        [Parameter(Mandatory=$true)][string[]]$AllowedMarkers
    )

    $Listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($Listener in $Listeners) {
        $Pid = [int]$Listener.OwningProcess
        $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $Pid" -ErrorAction SilentlyContinue
        if ($null -eq $Info) { continue }
        $CommandLine = [string]$Info.CommandLine
        $Owned = $false
        foreach ($Marker in $AllowedMarkers) {
            if ($CommandLine -like "*$Marker*") { $Owned = $true; break }
        }
        if (-not $Owned) {
            throw "TCP port $Port is already owned by non-MarketFusion PID $Pid. Command: $CommandLine"
        }
        Write-Host "Stopping stale MarketFusion listener on port $Port (PID $Pid)."
        Stop-Process -Id $Pid -Force -ErrorAction Stop
    }
}

Push-Location $Root
try {
    Stop-OwnedListener -Port 8765 -AllowedMarkers @('src.dashboard.full_api','src.dashboard.v07_api','uvicorn')
    Stop-OwnedListener -Port $DashboardPort -AllowedMarkers @('vite','dashboard')

    Remove-Item (Join-Path $PidRoot 'v07_api.json') -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $PidRoot 'v07_dashboard.json') -Force -ErrorAction SilentlyContinue

    $Launcher = Join-Path $Root 'scripts\start_marketfusion_dashboard.ps1'
    $Args = @('-Symbol', $Symbol, '-DashboardPort', [string]$DashboardPort)
    if ($SkipRuntime) { $Args += '-SkipRuntime' }
    if ($NoBrowser) { $Args += '-NoBrowser' }
    & $Launcher @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
