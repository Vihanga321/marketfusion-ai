param(
    [switch]$ConfirmArchive
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Forward = Join-Path $Root 'data\evaluation\XAUUSD\v10c_forward_validation'
$Models = Join-Path $Root 'models\XAUUSD\forward_validation\v10c'
$Runtime = Join-Path $Root 'data\runtime\v10c1'
$Preparation = Join-Path $Root 'data\evaluation\XAUUSD\v10c1_preparation.json'

if (-not $ConfirmArchive) {
    throw 'Refusing to archive the current V1.0C run without -ConfirmArchive. This operation preserves data but moves the old run out of the canonical forward paths.'
}

$Running = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like '*v10c_xauusd_forward_validation*' -or
    $_.CommandLine -like '*v10c1_reliability*'
})
if ($Running.Count -gt 0) {
    $Ids = ($Running | ForEach-Object { $_.ProcessId }) -join ', '
    throw "Stop the existing V1.0C/V1.0C.1 collector before preparation. Matching PIDs: $Ids"
}

$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$DataArchiveRoot = Join-Path $Root 'data\evaluation\XAUUSD\archive'
$ModelArchiveRoot = Join-Path $Root 'models\XAUUSD\forward_validation\archive'
New-Item -ItemType Directory -Force -Path $DataArchiveRoot, $ModelArchiveRoot | Out-Null

$DataArchive = $null
$ModelArchive = $null
if (Test-Path -LiteralPath $Forward) {
    $DataArchive = Join-Path $DataArchiveRoot "v10c_aborted_$Stamp"
    Move-Item -LiteralPath $Forward -Destination $DataArchive
    Write-Host "Archived previous forward run to: $DataArchive"
} else {
    Write-Host 'No existing V1.0C forward-data directory was present.'
}

if (Test-Path -LiteralPath $Models) {
    $ModelArchive = Join-Path $ModelArchiveRoot "v10c_aborted_$Stamp"
    Move-Item -LiteralPath $Models -Destination $ModelArchive
    Write-Host "Archived previous frozen model artifacts to: $ModelArchive"
} else {
    Write-Host 'No existing V1.0C frozen-model directory was present.'
}

if (Test-Path -LiteralPath $Runtime) {
    $RuntimeArchive = Join-Path $DataArchiveRoot "v10c1_runtime_$Stamp"
    Move-Item -LiteralPath $Runtime -Destination $RuntimeArchive
} else {
    $RuntimeArchive = $null
}

$Record = [ordered]@{
    contract_version = 'v1.0c1-clean-start-preparation-v1'
    prepared_at_utc = [DateTime]::UtcNow.ToString('o')
    previous_forward_archive = $DataArchive
    previous_models_archive = $ModelArchive
    previous_runtime_archive = $RuntimeArchive
    canonical_forward_path_clean = -not (Test-Path -LiteralPath $Forward)
    canonical_model_path_clean = -not (Test-Path -LiteralPath $Models)
    next_action = 'RUN_READ_ONLY_PREFLIGHT_THEN_INITIALIZE_EXACTLY_ONCE_AT_OFFICIAL_FORWARD_START'
    automatic_execution = 'DISABLED'
    runtime = 'SHADOW_ADVISORY_ONLY'
    manual_confirmation = 'REQUIRED'
    backfill = 'PROHIBITED'
}
$Record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Preparation -Encoding utf8

Write-Host ''
Write-Host 'V1.0C.1 CLEAN START PREPARATION: PASS'
Write-Host 'Old observations/models were preserved in archive paths; nothing was deleted.'
Write-Host 'DO NOT initialize until the official new forward start.'
Write-Host 'Next read-only check:'
Write-Host '  .\scripts\v10c1_preflight.ps1 -RequireUninitialized'
