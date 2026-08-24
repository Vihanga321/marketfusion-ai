$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Status = Join-Path $Root 'data\mt5\continuous\runtime_status.json'

if (-not (Test-Path -LiteralPath $Status)) {
    throw "No V0.5A runtime status yet: $Status"
}

$data = Get-Content -LiteralPath $Status -Raw | ConvertFrom-Json
Write-Host '=== MarketFusion V0.5A Status ==='
Write-Host "Status: $($data.status)"
Write-Host "Captured UTC: $($data.captured_at_utc)"
Write-Host "Symbol: $($data.symbol)"
Write-Host "Spread points: $($data.latest_spread_points)"
Write-Host "Dataset rows: $($data.dataset_rows)"
Write-Host "Feature-complete rows: $($data.feature_complete_rows)"
Write-Host "Daily history rows: $($data.daily_history_rows)"
Write-Host "Finalized daily rows: $($data.finalized_daily_history_rows)"
Write-Host "Latest day status: $($data.latest_daily_status)"
Write-Host "Last decision UTC: $($data.last_dataset_decision_utc)"
Write-Host "15m labels: $($data.matured_label_rows.'15m')"
Write-Host "60m labels: $($data.matured_label_rows.'60m')"
Write-Host "240m labels: $($data.matured_label_rows.'240m')"
Write-Host ''
foreach ($name in @('M1','M5','M15','H1')) {
    $tf = $data.timeframes.$name
    Write-Host ("{0}: rows={1} new={2} conflicts={3} last={4}" -f $name,$tf.rows_after,$tf.new_rows,$tf.conflict_rows,$tf.last_bar_close_utc)
}
