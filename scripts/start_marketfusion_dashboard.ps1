param(
    [ValidateSet('EURUSD','XAUUSD')][string]$Symbol = 'EURUSD',
    [switch]$SkipRuntime,
    [switch]$NoBrowser,
    [ValidateRange(1, 65535)][int]$DashboardPort = 4173
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$Dashboard = Join-Path $Root 'dashboard'
$NodeCommand = Get-Command node -ErrorAction SilentlyContinue
$Node = if ($null -eq $NodeCommand) { $null } else { $NodeCommand.Source }
$PidRoot = Join-Path $Root 'data\runtime\v07\pids'
$LogRoot = Join-Path $Root 'logs\v07'
$StartupTimeoutSeconds = 15
$DashboardUrl = "http://127.0.0.1:$DashboardPort"

function Get-MarketFusionProcessRecord {
    param([string]$PidFile)
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    try { return Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json } catch { return $null }
}

function Test-MarketFusionProcess {
    param([string]$PidFile, [string]$Marker)
    $Record = Get-MarketFusionProcessRecord $PidFile
    if ($null -eq $Record) { return $false }
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($Record.pid)" -ErrorAction SilentlyContinue
    return $null -ne $ProcessInfo -and $ProcessInfo.CommandLine -like "*$Marker*"
}

function Get-MarketFusionProcessTreeIds {
    param([int]$ProcessId)
    $Snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $Pending = @($ProcessId)
    $Ids = @()
    while ($Pending.Count -gt 0) {
        $ParentId = [int]$Pending[0]
        if ($Pending.Count -eq 1) { $Pending = @() } else { $Pending = @($Pending[1..($Pending.Count - 1)]) }
        if ($Ids -notcontains $ParentId) { $Ids += $ParentId }
        foreach ($Child in @($Snapshot | Where-Object { $_.ParentProcessId -eq $ParentId })) {
            if ($Ids -notcontains [int]$Child.ProcessId) { $Pending += [int]$Child.ProcessId }
        }
    }
    return @($Ids)
}

function Test-MarketFusionPortOwner {
    param([int]$Port, [int]$ProcessId, [string]$Marker)
    $ProcessIds = @(Get-MarketFusionProcessTreeIds $ProcessId)
    $Listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalAddress -in @('127.0.0.1', '0.0.0.0') })
    foreach ($Listener in $Listeners) {
        if ($ProcessIds -notcontains [int]$Listener.OwningProcess) { continue }
        $Owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($Listener.OwningProcess)" -ErrorAction SilentlyContinue
        if ($null -ne $Owner -and $Owner.CommandLine -like "*$Marker*") { return $true }
    }
    return $false
}

function Stop-FailedMarketFusionProcess {
    param([string]$PidFile, [string]$Marker, [int]$ExpectedPid)
    $Record = Get-MarketFusionProcessRecord $PidFile
    if ($null -eq $Record -or [int]$Record.pid -ne $ExpectedPid) { return }
    if (Test-MarketFusionProcess $PidFile $Marker) {
        $Ids = @(Get-MarketFusionProcessTreeIds $ExpectedPid)
        [array]::Reverse($Ids)
        foreach ($Id in $Ids) {
            $Info = Get-CimInstance Win32_Process -Filter "ProcessId = $Id" -ErrorAction SilentlyContinue
            if ($null -ne $Info -and $Info.CommandLine -like "*$Marker*") {
                Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

function Wait-MarketFusionEndpoint {
    param(
        [string]$Name,
        [string]$Marker,
        [int]$ProcessId,
        [int]$Port,
        [string]$Url,
        [string]$ErrLog,
        [string]$ExpectedCorsOrigin = ''
    )
    $PidFile = Join-Path $PidRoot "$Name.json"
    $Deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $LastHttpError = $null
    while ([DateTime]::UtcNow -lt $Deadline) {
        if (-not (Test-MarketFusionProcess $PidFile $Marker)) { break }
        if (Test-MarketFusionPortOwner $Port $ProcessId $Marker) {
            try {
                $Headers = if ($ExpectedCorsOrigin) { @{ Origin = $ExpectedCorsOrigin } } else { @{} }
                $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -Headers $Headers -TimeoutSec 2
                $HttpSuccess = [int]$Response.StatusCode -ge 200 -and [int]$Response.StatusCode -lt 400
                $CorsSuccess = -not $ExpectedCorsOrigin -or [string]$Response.Headers['Access-Control-Allow-Origin'] -eq $ExpectedCorsOrigin
                if ($HttpSuccess -and $CorsSuccess) { return }
                if (-not $CorsSuccess) { $LastHttpError = "API did not allow configured dashboard origin $ExpectedCorsOrigin" }
            } catch { $LastHttpError = $_.Exception.Message }
        }
        Start-Sleep -Milliseconds 250
    }
    Stop-FailedMarketFusionProcess $PidFile $Marker $ProcessId
    $Message = "$Name failed readiness validation within $StartupTimeoutSeconds seconds. Expected PID $ProcessId to own TCP 127.0.0.1:$Port and $Url to return HTTP success. Failed PID record removed."
    if ($LastHttpError) { $Message += " Last HTTP error: $LastHttpError" }
    if (Test-Path -LiteralPath $ErrLog) {
        $LatestError = (Get-Content -Tail 30 -LiteralPath $ErrLog -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        if ($LatestError) { $Message += "`nLatest stderr ($ErrLog):`n$LatestError" }
        else { $Message += " See stderr log: $ErrLog" }
    }
    throw $Message
}

function Start-MarketFusionProcess {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$Marker)
    $PidFile = Join-Path $PidRoot "$Name.json"
    if (Test-MarketFusionProcess $PidFile $Marker) {
        $Record = Get-MarketFusionProcessRecord $PidFile
        Write-Host "$Name already running; duplicate start skipped."
        return [int]$Record.pid
    }
    if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
    $OutLog = Join-Path $LogRoot "$Name.out.log"; $ErrLog = Join-Path $LogRoot "$Name.err.log"
    $Process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    @{ pid = $Process.Id; name = $Name; marker = $Marker; started_at_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding utf8
    Write-Host "Started $Name as PID $($Process.Id)."
    return [int]$Process.Id
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
$HadDashboardPortEnvironment = Test-Path Env:\MARKETFUSION_DASHBOARD_PORT
$PreviousDashboardPortEnvironment = $env:MARKETFUSION_DASHBOARD_PORT
$HadSymbolEnvironment = Test-Path Env:\MARKETFUSION_ACTIVE_SYMBOL
$PreviousSymbolEnvironment = $env:MARKETFUSION_ACTIVE_SYMBOL
$HadViteSymbolEnvironment = Test-Path Env:\VITE_MARKETFUSION_SYMBOL
$PreviousViteSymbolEnvironment = $env:VITE_MARKETFUSION_SYMBOL
$env:MARKETFUSION_DASHBOARD_PORT = [string]$DashboardPort
$env:MARKETFUSION_ACTIVE_SYMBOL = $Symbol
$env:VITE_MARKETFUSION_SYMBOL = $Symbol
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    if (-not $Node) { throw 'Node.js is not installed or not on PATH' }
    if (-not (Test-Path -LiteralPath (Join-Path $Dashboard 'node_modules\vite\bin\vite.js'))) { Push-Location $Dashboard; try { npm ci; if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' } } finally { Pop-Location } }
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }
    New-Item -ItemType Directory -Force -Path $PidRoot, $LogRoot | Out-Null
    if (-not $SkipRuntime) {
        $QuoteName = "realtime_quote_$($Symbol.ToLowerInvariant())"
        $QuotePid = Start-MarketFusionProcess $QuoteName $Python @('-m','src.marketdata.realtime_quote','--symbol',$Symbol,'--poll-ms','250') 'src.marketdata.realtime_quote'
        Write-Host "MARKETFUSION REALTIME QUOTE: STARTED (PID $QuotePid, 250ms poll, display-only)"
        if ($Symbol -eq 'XAUUSD') { $null = Start-MarketFusionProcess 'v10a_xauusd_state' $Python @('-m','src.runtime.v10a_asset_state','--symbol','XAUUSD','--continuous','--interval-seconds','15') 'src.runtime.v10a_asset_state' }
        elseif (Test-LegacyMarketFusionProcess 'v06_runtime' 'src.runtime.v06c_runner') { Write-Host 'v06_runtime already running from the V0.6 launcher; duplicate start skipped.' }
        else { $null = Start-MarketFusionProcess 'v06_runtime' $Python @('-m','src.runtime.v06c_runner','--continuous','--interval-seconds','15') 'src.runtime.v06c_runner' }
    }
    $ApiErrLog = Join-Path $LogRoot 'v07_api.err.log'
    $ApiPid = Start-MarketFusionProcess 'v07_api' $Python @('-m','src.dashboard.full_api','--host','127.0.0.1','--port','8765') 'src.dashboard.full_api'
    Wait-MarketFusionEndpoint 'v07_api' 'src.dashboard.full_api' $ApiPid 8765 "http://127.0.0.1:8765/api/health?symbol=$Symbol" $ApiErrLog $DashboardUrl
    Write-Host 'MARKETFUSION FULL API: PASS'
    Write-Host "API PID: $ApiPid"
    Write-Host 'API port: 8765 LISTEN'

    $DashboardErrLog = Join-Path $LogRoot 'v07_dashboard.err.log'
    $DashboardPid = Start-MarketFusionProcess 'v07_dashboard' $Node @('dashboard/node_modules/vite/bin/vite.js','dashboard','--host','127.0.0.1','--port',[string]$DashboardPort,'--strictPort') 'vite/bin/vite.js'
    Wait-MarketFusionEndpoint 'v07_dashboard' 'vite/bin/vite.js' $DashboardPid $DashboardPort $DashboardUrl $DashboardErrLog
    Write-Host 'MARKETFUSION FULL DASHBOARD: PASS'
    Write-Host "Dashboard PID: $DashboardPid"
    Write-Host "Dashboard port: $DashboardPort LISTEN"
    Write-Host "Dashboard URL: $DashboardUrl"
    Write-Host 'Live quote path is display-only; causal model inputs remain completed candles.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'
    if (-not $NoBrowser) { Start-Process $DashboardUrl }
} finally {
    if ($HadDashboardPortEnvironment) { $env:MARKETFUSION_DASHBOARD_PORT = $PreviousDashboardPortEnvironment }
    else { Remove-Item Env:\MARKETFUSION_DASHBOARD_PORT -ErrorAction SilentlyContinue }
    if ($HadSymbolEnvironment) { $env:MARKETFUSION_ACTIVE_SYMBOL = $PreviousSymbolEnvironment }
    else { Remove-Item Env:\MARKETFUSION_ACTIVE_SYMBOL -ErrorAction SilentlyContinue }
    if ($HadViteSymbolEnvironment) { $env:VITE_MARKETFUSION_SYMBOL = $PreviousViteSymbolEnvironment }
    else { Remove-Item Env:\VITE_MARKETFUSION_SYMBOL -ErrorAction SilentlyContinue }
    Pop-Location
}
