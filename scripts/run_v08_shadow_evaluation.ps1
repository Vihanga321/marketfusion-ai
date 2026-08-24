$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_v08_once.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot 'show_v08_status.ps1')
