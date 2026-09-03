param(
    [ValidateRange(1, 90)][int]$KeepDays = 14
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Forward = Join-Path $Root 'data\evaluation\XAUUSD\v10c_forward_validation'
$Runtime = Join-Path $Root 'data\runtime\v10c1'
$BackupRoot = Join-Path $Root 'data\backups\v10c1'

if (-not (Test-Path -LiteralPath (Join-Path $Forward 'manifest.json'))) {
    throw 'Forward validation is not initialized; there is no V1.0C.1 state to back up.'
}

$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$Destination = Join-Path $BackupRoot $Stamp
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Files = @(
    (Join-Path $Forward 'manifest.json'),
    (Join-Path $Forward 'latest_status.json'),
    (Join-Path $Forward 'predictions.parquet'),
    (Join-Path $Forward 'outcomes.parquet'),
    (Join-Path $Runtime 'collector_health.json'),
    (Join-Path $Runtime 'provider_outages.jsonl')
)
$Inventory = @()
foreach ($File in $Files) {
    if (-not (Test-Path -LiteralPath $File)) { continue }
    $Name = Split-Path -Leaf $File
    $Target = Join-Path $Destination $Name
    Copy-Item -LiteralPath $File -Destination $Target
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()
    $Info = Get-Item -LiteralPath $Target
    $Inventory += [ordered]@{
        name = $Name
        bytes = $Info.Length
        sha256 = $Hash
    }
}

$Manifest = [ordered]@{
    contract_version = 'v1.0c1-backup-v1'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    source_forward_path = $Forward
    automatic_execution = 'DISABLED'
    runtime = 'SHADOW_ADVISORY_ONLY'
    files = $Inventory
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Destination 'backup_manifest.json') -Encoding utf8

$Cutoff = [DateTime]::UtcNow.AddDays(-$KeepDays)
Get-ChildItem -LiteralPath $BackupRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
    $_.CreationTimeUtc -lt $Cutoff
} | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "V1.0C.1 backup created: $Destination"
Write-Host "Files copied: $($Inventory.Count)"
Write-Host 'Automatic execution: DISABLED'
