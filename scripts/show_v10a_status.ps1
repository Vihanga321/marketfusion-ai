$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $repo 'data\backfill\v10a\latest_status.json'
if (-not (Test-Path -LiteralPath $statusPath)) {
    Write-Host 'No V1.0A local status exists yet. Run .\scripts\run_v10a_audit.ps1 first.'
    exit 0
}
$status = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
Write-Host 'MARKETFUSION V1.0A AUDITED MT5 BACKFILL'
Write-Host "status: $($status.status)"
Write-Host "mode: $($status.mode)"
Write-Host "applied: $($status.applied)"
Write-Host "bar_overlap_conflicts: $($status.bar_overlap_conflicts)"
Write-Host "dataset_overlap_mismatches: $($status.dataset_overlap_mismatches)"
Write-Host "eligible_rows: $($status.market_core.eligible_rows)"
Write-Host "eligible_trading_days: $($status.market_core.eligible_trading_days)"
Write-Host "calendar_span_days: $($status.market_core.calendar_span_days)"
Write-Host "promotion_history_gate: $($status.market_core.promotion_history_gate)"
Write-Host "automatic_v05c_training: $($status.automatic_v05c_training)"
Write-Host "automatic_model_promotion: $($status.automatic_model_promotion)"
Write-Host 'trading: DISABLED'
