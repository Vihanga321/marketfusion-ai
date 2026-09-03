param(
    [switch]$ConfirmInstall,
    [ValidateRange(30, 3600)][int]$IntervalSeconds = 60
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $IsWindows) { throw 'This startup-task installer is supported on Windows only.' }
if (-not $ConfirmInstall) { throw 'Use -ConfirmInstall to create the Windows logon recovery task.' }

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Manifest = Join-Path $Root 'data\evaluation\XAUUSD\v10c_forward_validation\manifest.json'
$Launcher = Join-Path $Root 'scripts\run_v10c1_reliable_collector.ps1'
$TaskName = 'MarketFusion-V10C1-Collector'

if (-not (Test-Path -LiteralPath $Manifest)) {
    throw 'V1.0C.1 forward validation is not initialized. Do not install automatic collector startup before the official one-time initialization.'
}
$State = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
if ($State.automatic_execution -ne 'DISABLED' -or $State.runtime -ne 'SHADOW_ADVISORY_ONLY' -or $State.manual_confirmation -ne 'REQUIRED') {
    throw 'Frozen manifest safety invariants are not valid; startup task not installed.'
}

$PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Launcher`" -IntervalSeconds $IntervalSeconds"
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $Root
$UserId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description 'MarketFusion V1.0C.1 XAUUSD frozen forward reliability collector. No trading execution.'
Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null

Write-Host "Installed Windows task: $TaskName"
Write-Host 'Trigger: current-user logon'
Write-Host 'Recovery: restart after unexpected exit'
Write-Host 'Single-instance protection remains active.'
Write-Host 'Automatic execution: DISABLED'
