param(
    [switch]$NoBrowser,
    [ValidateRange(1, 65535)][int]$DashboardPort = 4173
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$PidRoot = Join-Path $Root 'data\runtime\v07\pids'
$LogRoot = Join-Path $Root 'logs\v07'

function Start-CollectorIfMissing {
    param([string]$Name, [string[]]$Arguments, [string]$Marker)
    $PidFile = Join-Path $PidRoot "$Name.json"
    $LegacyPidFile = Join-Path $Root "data\runtime\v06\pids\$Name.pid"
    if (Test-Path -LiteralPath $LegacyPidFile) {
        $LegacyPid = 0
        if ([int]::TryParse((Get-Content -Raw -LiteralPath $LegacyPidFile).Trim(), [ref]$LegacyPid)) {
            $LegacyInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $LegacyPid" -ErrorAction SilentlyContinue
            if ($null -ne $LegacyInfo -and $LegacyInfo.CommandLine -like "*$Marker*") { Write-Host "$Name already running from the V0.6 launcher; duplicate start skipped."; return }
        }
    }
    if (Test-Path -LiteralPath $PidFile) {
        try { $Record = Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json; $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $($Record.pid)" -ErrorAction SilentlyContinue } catch { $Info = $null }
        if ($null -ne $Info -and $Info.CommandLine -like "*$Marker*") { Write-Host "$Name already running; duplicate start skipped."; return }
        Remove-Item -LiteralPath $PidFile -Force
    }
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogRoot "$Name.out.log") -RedirectStandardError (Join-Path $LogRoot "$Name.err.log")
    @{ pid=$Process.Id; name=$Name; marker=$Marker; started_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding utf8
    Write-Host "Started $Name as PID $($Process.Id)."
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }
} finally { Pop-Location }
New-Item -ItemType Directory -Force -Path $PidRoot, $LogRoot | Out-Null
Start-CollectorIfMissing 'v05a_collector' @('-m','src.marketdata.v05a_runner','--poll-seconds','15','--overlap-bars','8') 'src.marketdata.v05a_runner'
Start-CollectorIfMissing 'v05b_collector' @('-m','src.intelligence.v05b_runner','--interval','300') 'src.intelligence.v05b_runner'
Write-Host 'V0.5C continuous training is intentionally NOT started.'
& (Join-Path $PSScriptRoot 'start_marketfusion_dashboard.ps1') -NoBrowser:$NoBrowser -DashboardPort $DashboardPort
Start-CollectorIfMissing 'v08_monitor' @('-m','src.evaluation.v08_runner','--continuous','--interval-seconds','300') 'src.evaluation.v08_runner'
Write-Host 'V0.8 heavy model research is intentionally NOT started; run it manually or on a controlled weekly schedule.'
