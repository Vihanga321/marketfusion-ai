$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Status = Join-Path $Root 'data\intelligence\runtime_status.json'
if (-not (Test-Path -LiteralPath $Status)) { throw "No V0.5B runtime status yet: $Status" }
$data = Get-Content -LiteralPath $Status -Raw | ConvertFrom-Json
Write-Host '=== MarketFusion V0.5B Status ==='
Write-Host "Status: $($data.status)"
Write-Host "Captured UTC: $($data.captured_at_utc)"
Write-Host "News rows: $($data.news_rows_total) (new $($data.new_news_rows))"
Write-Host "Macro rows: $($data.macro_rows_total) (new $($data.new_macro_rows))"
Write-Host "Context rows: $($data.context_rows_total)"
Write-Host "News 15m/60m/4h/24h: $($data.current_news_15m) / $($data.current_news_60m) / $($data.current_news_240m) / $($data.current_news_1440m)"
Write-Host "Fed funds: $($data.macro_us_effective_fed_funds_value)"
Write-Host "US 2Y: $($data.macro_us_2y_yield_value)"
Write-Host "US 10Y: $($data.macro_us_10y_yield_value)"
Write-Host "US 10Y-2Y: $($data.macro_us_10y_minus_2y_value)"
Write-Host "ECB MRR: $($data.macro_ecb_main_refinancing_rate_value)"
Write-Host "Policy spread US-ECB: $($data.macro_policy_rate_spread_us_minus_ecb_pctpt)"
Write-Host "News source errors: $($data.news_errors.Count)"
Write-Host "Macro source errors: $($data.macro_errors.Count)"
Write-Host "Trading: $($data.trading)"
