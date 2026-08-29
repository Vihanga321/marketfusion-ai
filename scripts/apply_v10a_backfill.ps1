$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Test-MarketFusionV05AProcess {
    $jsonPid = Join-Path $repo 'data\runtime\v07\pids\v05a_collector.json'
    if (Test-Path -LiteralPath $jsonPid) {
        try {
            $record = Get-Content -Raw -LiteralPath $jsonPid | ConvertFrom-Json
            $info = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
            if ($null -ne $info -and $info.CommandLine -like '*src.marketdata.v05a_runner*') { return $true }
        } catch {}
    }
    $legacyPid = Join-Path $repo 'data\runtime\v06\pids\v05a_collector.pid'
    if (Test-Path -LiteralPath $legacyPid) {
        $pidValue = 0
        if ([int]::TryParse((Get-Content -Raw -LiteralPath $legacyPid).Trim(), [ref]$pidValue)) {
            $info = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
            if ($null -ne $info -and $info.CommandLine -like '*src.marketdata.v05a_runner*') { return $true }
        }
    }
    return $false
}

if (Test-MarketFusionV05AProcess) {
    throw 'V0.5A continuous collector is running. Run .\scripts\stop_marketfusion.ps1 first, leave MT5 itself open, then retry V1.0A apply.'
}

Write-Host '=== MarketFusion V1.0A APPLY Audited Historical Backfill ==='
Write-Host 'MT5 must remain open. Existing V0.5A files will be backed up and rollback is automatic on failure.'
python -m src.backfill.v10a_runner --apply
if ($LASTEXITCODE -ne 0) { throw 'V1.0A apply failed' }
Write-Host '=== V1.0A APPLY COMPLETE ==='
