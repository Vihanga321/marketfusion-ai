param([switch]$SkipRuntime, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$Dashboard = Join-Path $Root 'dashboard'
$NodeCommand = Get-Command node -ErrorAction SilentlyContinue
$Node = if ($null -eq $NodeCommand) { $null } else { $NodeCommand.Source }
$PidRoot = Join-Path $Root 'data\runtime\v07\pids'
$LogRoot = Join-Path $Root 'logs\v07'

function Test-MarketFusionProcess {
    param([string]$PidFile, [string]$Marker)
    if (-not (Test-Path -LiteralPath $PidFile)) { return $false }
    try { $Record = Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json } catch { return $false }
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($Record.pid)" -ErrorAction SilentlyContinue
    return $null -ne $ProcessInfo -and $ProcessInfo.CommandLine -like "*$Marker*"
}

function Start-MarketFusionProcess {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$Marker)
    $PidFile = Join-Path $PidRoot "$Name.json"
    if (Test-MarketFusionProcess $PidFile $Marker) { Write-Host "$Name already running; duplicate start skipped."; return }
    if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
    $OutLog = Join-Path $LogRoot "$Name.out.log"; $ErrLog = Join-Path $LogRoot "$Name.err.log"
    $Process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    @{ pid = $Process.Id; name = $Name; marker = $Marker; started_at_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding utf8
    Write-Host "Started $Name as PID $($Process.Id)."
}

function Test-LegacyMarketFusionProcess {
    param([string]$Name, [string]$Marker)
    $LegacyPidFile = Join-Path $Root "data\runtime\v06\pids\$Name.pid"
    if (-not (Test-Path -LiteralPath $LegacyPidFile)) { return $false }
    $LegacyPid = 0
    if (-not [int]::TryParse((Get-Content -Raw -LiteralPath $LegacyPidFile).Trim(), [ref]$LegacyPid)) { return $false }
    $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $LegacyPid" -ErrorAction SilentlyContinue
    return $null -ne $Info -and $Info.CommandLine -like "*$Marker*"
}

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    if (-not $Node) { throw 'Node.js is not installed or not on PATH' }
    if (-not (Test-Path -LiteralPath (Join-Path $Dashboard 'node_modules\vite\bin\vite.js'))) { Push-Location $Dashboard; try { npm ci; if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' } } finally { Pop-Location } }
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }
    New-Item -ItemType Directory -Force -Path $PidRoot, $LogRoot | Out-Null
    if (-not $SkipRuntime) {
        if (Test-LegacyMarketFusionProcess 'v06_runtime' 'src.runtime.v06c_runner') { Write-Host 'v06_runtime already running from the V0.6 launcher; duplicate start skipped.' }
        else { Start-MarketFusionProcess 'v06_runtime' $Python @('-m','src.runtime.v06c_runner','--continuous','--interval-seconds','15') 'src.runtime.v06c_runner' }
    }
    Start-MarketFusionProcess 'v07_api' $Python @('-m','src.dashboard.v07_api','--host','127.0.0.1','--port','8765') 'src.dashboard.v07_api'
    Start-MarketFusionProcess 'v07_dashboard' $Node @('dashboard/node_modules/vite/bin/vite.js','dashboard','--host','127.0.0.1','--port','5173','--strictPort') 'vite/bin/vite.js'
    Start-Sleep -Seconds 2
    Write-Host 'MarketFusion dashboard: http://127.0.0.1:5173'
    Write-Host 'Trading execution remains disabled; manual execution in MT5 only.'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }
} finally { Pop-Location }
