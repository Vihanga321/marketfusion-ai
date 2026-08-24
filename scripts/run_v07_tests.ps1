$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
Push-Location $Root
try {
    & $Python -m unittest tests.test_v07_dashboard_api -v
    if ($LASTEXITCODE -ne 0) { throw 'V0.7 API tests failed' }
    Push-Location dashboard
    try {
        npm run typecheck; if ($LASTEXITCODE -ne 0) { throw 'V0.7 typecheck failed' }
        npm run test; if ($LASTEXITCODE -ne 0) { throw 'V0.7 frontend tests failed' }
        npm run build; if ($LASTEXITCODE -ne 0) { throw 'V0.7 frontend build failed' }
    } finally { Pop-Location }
    Write-Host 'V07_TEST_STATUS: PASS'
} finally { Pop-Location }
