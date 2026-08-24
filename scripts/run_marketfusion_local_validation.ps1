param(
    [switch]$SkipProviderProbe
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'

function Invoke-MarketFusionStage {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "`n=== $Name ==="
    try {
        & $Action | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "$Name exited with code $LASTEXITCODE"
        }
        return 'PASS'
    } catch {
        Write-Warning "$Name failed: $($_.Exception.Message)"
        return 'FAIL'
    }
}

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }

    Write-Host 'MARKETFUSION FULL LOCAL RUNTIME VALIDATION'
    Write-Host 'Read-only local research validation. Trading execution is forbidden.'

    $RepoSafety = Invoke-MarketFusionStage '1. Repository safety' { & $Python scripts\repo_safety_check.py }
    $V05ATests = Invoke-MarketFusionStage '2. V0.5A deterministic tests' { & (Join-Path $PSScriptRoot 'run_v05a_tests.ps1') }
    $V05BTests = Invoke-MarketFusionStage '3. V0.5B deterministic tests' { & (Join-Path $PSScriptRoot 'run_v05b_tests.ps1') }
    $Mt5 = Invoke-MarketFusionStage '4. Read-only MT5 connectivity' { & $Python scripts\local_runtime_validation.py mt5 }
    $V05ACycle = Invoke-MarketFusionStage '5. V0.5A one-cycle validation' { & (Join-Path $PSScriptRoot 'run_v05a_once.ps1') }

    $Snapshot = Get-ChildItem "$env:APPDATA\MetaQuotes\Terminal" `
        -Filter 'marketfusion_mt5_target_snapshots.tsv' -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $Snapshot) {
        Write-Warning '6. V0.4D skipped: no MT5 target snapshot file was found.'
        $V04D = 'SKIPPED'
    } else {
        $SnapshotPath = $Snapshot.FullName
        $V04D = Invoke-MarketFusionStage '6. V0.4D live consensus and surprise audit' {
            & (Join-Path $PSScriptRoot 'run_v04d_mt5_live_surprise_pipeline.ps1') -SnapshotsTsv $SnapshotPath
        }
    }

    if (-not $SkipProviderProbe) {
        $null = Invoke-MarketFusionStage '7a. V0.5B independent free-source diagnostics' {
            & $Python scripts\local_runtime_validation.py providers
        }
    } else {
        Write-Host "`n=== 7a. V0.5B independent free-source diagnostics: SKIPPED BY REQUEST ==="
    }
    $V05BCycle = Invoke-MarketFusionStage '7b. V0.5B one-cycle validation' { & (Join-Path $PSScriptRoot 'run_v05b_once.ps1') }

    Write-Host "`n=== 8. V0.5A + V0.5B integration and final audit ==="
    & $Python scripts\local_runtime_validation.py audit `
        --repo-safety $RepoSafety `
        --v05a-tests $V05ATests `
        --v05b-tests $V05BTests `
        --mt5 $Mt5 `
        --v05a-cycle $V05ACycle `
        --v04d $V04D `
        --v05b-cycle $V05BCycle
    $AuditExit = $LASTEXITCODE

    Write-Host "`n=== 9. Final summary ==="
    Get-Content -LiteralPath reports\marketfusion_full_local_runtime_validation.txt
    if ($AuditExit -ne 0) {
        throw "Master local validation failed with code $AuditExit"
    }
} finally {
    Pop-Location
}
