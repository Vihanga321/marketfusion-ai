$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Status = Join-Path $Root 'data\evaluation\v08\latest_status.json'
if (-not (Test-Path -LiteralPath $Status)) { Write-Host 'V08_STATUS: INSUFFICIENT_DATA (no monitor cycle yet)'; exit 0 }
$Payload = Get-Content -Raw -LiteralPath $Status | ConvertFrom-Json
Write-Host "V08_STATUS: $($Payload.status)"
Write-Host "predictions: $($Payload.performance.recorded_predictions)"
Write-Host "matured outcomes: $($Payload.performance.matured_outcomes)"
Write-Host "directional calls: $((@($Payload.performance.horizons.PSObject.Properties.Value | ForEach-Object { $_.directional_calls }) | Measure-Object -Sum).Sum)"
Write-Host "calibration: $($Payload.performance.calibration_status)"
Write-Host "market drift: $($Payload.performance.market_drift_status)"
Write-Host "model drift: $($Payload.performance.model_drift_status)"
Write-Host "champion 15m: $($Payload.champions.'15')"
Write-Host "champion 60m: $($Payload.champions.'60')"
Write-Host "champion 240m: $($Payload.champions.'240')"
Write-Host 'trading execution: DISABLED'
