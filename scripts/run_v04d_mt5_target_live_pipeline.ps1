param(
    [string]$SnapshotsTsv = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D Target Live Consensus Pipeline ==='

    if ([string]::IsNullOrWhiteSpace($SnapshotsTsv)) {
        $candidate = Get-ChildItem "$env:APPDATA\MetaQuotes\Terminal" `
            -Filter 'marketfusion_mt5_target_snapshots.tsv' `
            -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -eq $candidate) {
            throw 'Could not find marketfusion_mt5_target_snapshots.tsv under the MT5 terminal folders.'
        }
        $SnapshotsTsv = $candidate.FullName
    }

    $ResolvedSnapshots = (Resolve-Path -LiteralPath $SnapshotsTsv).Path
    Write-Host "Target snapshot file: $ResolvedSnapshots"

    Write-Host '[1/2] Import and audit target snapshots'
    & (Join-Path $PSScriptRoot 'run_v04d_mt5_calendar_audit.ps1') -SnapshotsTsv $ResolvedSnapshots
    if ($LASTEXITCODE -ne 0) { throw 'Target snapshot import/audit failed' }

    Write-Host '[2/2] Run strict live consensus gate'
    & (Join-Path $PSScriptRoot 'run_v04d_mt5_live_consensus_gate.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Target live consensus gate failed' }

    $Canonical = Join-Path $Root 'data\mt5\calendar\v04d_live_verified_consensus.csv'
    if (-not (Test-Path -LiteralPath $Canonical)) {
        throw "Missing canonical verified consensus output: $Canonical"
    }

    $rows = @(Import-Csv -LiteralPath $Canonical)
    Write-Host ''
    Write-Host "Canonical verified consensus rows: $($rows.Count)"
    if ($rows.Count -gt 0) {
        $rows |
            Select-Object event_id,event_name,event_timestamp_utc,forecast_value,captured_at_gmt,consensus_status |
            Format-Table -AutoSize
    } else {
        Write-Host 'No target row is model-eligible yet. Keep the MT5 target collector running and rerun this pipeline later.'
    }

    Write-Host '=== V0.4D TARGET LIVE CONSENSUS PIPELINE COMPLETE ==='
} finally {
    Pop-Location
}
