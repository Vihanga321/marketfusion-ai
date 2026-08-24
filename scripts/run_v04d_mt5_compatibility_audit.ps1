$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

Push-Location $Root
try {
    Write-Host '=== MarketFusion V0.4D MT5 Measure Compatibility Audit ==='
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }
    if (-not (Test-Path -LiteralPath 'reports\v04d_mt5_v04c_actual_comparison.csv')) {
        throw 'Missing reports\v04d_mt5_v04c_actual_comparison.csv. Run the identity audit first.'
    }

    Write-Host '[1/3] Compile compatibility-audit code'
    & $Python -m compileall -q src\events\mt5_calendar_compatibility_audit.py tests\test_v04d_mt5_compatibility.py
    if ($LASTEXITCODE -ne 0) { throw 'Compatibility audit Python compile failed' }

    Write-Host '[2/3] Run compatibility-audit tests'
    & $Python -m unittest tests.test_v04d_mt5_compatibility -v
    if ($LASTEXITCODE -ne 0) { throw 'Compatibility audit tests failed' }

    Write-Host '[3/3] Audit exact vs provider-precision compatibility'
    & $Python src\events\mt5_calendar_compatibility_audit.py
    if ($LASTEXITCODE -ne 0) { throw 'MT5 measure compatibility audit failed' }

    Write-Host '=== V0.4D MT5 MEASURE COMPATIBILITY AUDIT COMPLETE (FAIL-CLOSED) ==='
    Write-Host 'Historical MT5 forecasts remain non-model-eligible until original pre-release timing is proven.'
    Write-Host 'Future verified surprises should prefer provider-consistent MT5 forecast + MT5 actual, with official actuals retained as an independent validation layer.'
} finally {
    Pop-Location
}
