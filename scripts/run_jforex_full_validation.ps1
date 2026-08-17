param(
    [int]$BatchSize = 12,
    [int]$RetryIncompletePasses = 1
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$BatchRoot = Join-Path $Root 'data\dukascopy\full_validation_batches'
$AllEvents = Join-Path $Root 'data\dukascopy\events_for_jforex.tsv'
$EventParquet = Join-Path $Root 'data\events\bls_release_events.parquet'
$JForexRoot = Join-Path $Root 'jforex-event-exporter'
$ValidatorClass = 'ai.marketfusion.jforex.TickValidationBatchRobustV2'

function Get-TsvRows {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing TSV: $Path" }
    return @(Import-Csv -LiteralPath $Path -Delimiter "`t")
}

function Remove-SafeWorkDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BatchDirectory
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $ResolvedBatch = [System.IO.Path]::GetFullPath($BatchDirectory).TrimEnd('\') + '\'
    $ResolvedTarget = [System.IO.Path]::GetFullPath($Path).TrimEnd('\') + '\'
    if ($ResolvedTarget -eq $ResolvedBatch -or
        -not $ResolvedTarget.StartsWith($ResolvedBatch, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove work directory outside batch: $Path"
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Test-BatchOutputs {
    param(
        [Parameter(Mandatory = $true)][string]$InputFile,
        [Parameter(Mandatory = $true)][string]$ResultDir,
        [Parameter(Mandatory = $true)][string]$BatchName
    )
    $Summary = Join-Path $ResultDir 'robust_tick_validation_summary.tsv'
    $Chunks = Join-Path $ResultDir 'robust_tick_chunk_status.tsv'
    $Metadata = Join-Path $ResultDir 'result_metadata.json'
    if (-not (Test-Path -LiteralPath $Summary) -or -not (Test-Path -LiteralPath $Chunks) -or -not (Test-Path -LiteralPath $Metadata)) {
        throw "Canonical validator outputs/metadata are missing for $BatchName"
    }
    $ExpectedRows = Get-TsvRows -Path $InputFile
    $SummaryRows = Get-TsvRows -Path $Summary
    if ($SummaryRows.Count -ne $ExpectedRows.Count) {
        throw "Batch output row-count mismatch for ${BatchName}: expected $($ExpectedRows.Count), got $($SummaryRows.Count)"
    }
    $ExpectedIds = @($ExpectedRows | ForEach-Object { $_.event_id } | Sort-Object)
    $ActualIds = @($SummaryRows | ForEach-Object { $_.event_id } | Sort-Object)
    if (($ExpectedIds -join "`n") -ne ($ActualIds -join "`n")) {
        throw "Batch output event-set mismatch for $BatchName"
    }
    $BadStatuses = @($SummaryRows | Where-Object { $_.status -notin @('PASS', 'INCOMPLETE', 'MISMATCH', 'ERROR') })
    if ($BadStatuses.Count -gt 0) { throw "Unknown validation status in $BatchName" }
    return $SummaryRows
}

function Invoke-ValidatorAttempt {
    param(
        [Parameter(Mandatory = $true)][System.IO.DirectoryInfo]$Batch,
        [Parameter(Mandatory = $true)][string]$RuntimeClasspath,
        [Parameter(Mandatory = $true)][string]$JavaExecutable,
        [Parameter(Mandatory = $true)][int]$Ordinal,
        [Parameter(Mandatory = $true)][int]$Total
    )
    $InputFile = Join-Path $Batch.FullName 'events.tsv'
    $ResultDir = Join-Path $Batch.FullName 'result'
    $AttemptInput = Join-Path $Batch.FullName '.attempt_events.tsv'
    $AttemptResult = Join-Path $Batch.FullName '.attempt_result'
    $PlanFile = Join-Path $Batch.FullName '.resume_plan.json'

    for ($PassNumber = 0; $PassNumber -le $RetryIncompletePasses; $PassNumber++) {
        & $Python src\events\resume_jforex_validation.py plan `
            --input $InputFile --result-dir $ResultDir `
            --attempt-input $AttemptInput --plan-file $PlanFile
        if ($LASTEXITCODE -ne 0) { throw "Resume planning failed for $($Batch.Name)" }
        $Plan = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json
        if ($Plan.mode -eq 'SKIP_PASS') {
            Write-Host "$($Batch.Name): compatible PASS result reused"
            break
        }
        if ($Plan.mode -eq 'SKIP_HARD_FAILURE') {
            Write-Warning "$($Batch.Name): compatible MISMATCH/ERROR result preserved for final failure report"
            break
        }

        Remove-SafeWorkDirectory -Path $AttemptResult -BatchDirectory $Batch.FullName
        New-Item -ItemType Directory -Force -Path $AttemptResult | Out-Null
        $AttemptNumber = $PassNumber + 1
        $MaximumAttempts = $RetryIncompletePasses + 1
        Write-Host ("--- {0} ({1}/{2}) validation pass {3}/{4}; events={5} ---" -f `
            $Batch.Name, $Ordinal, $Total, $AttemptNumber, $MaximumAttempts, @($Plan.retry_event_ids).Count)

        & $JavaExecutable -cp $RuntimeClasspath $ValidatorClass $AttemptInput $AttemptResult
        $JavaExit = $LASTEXITCODE
        if ($JavaExit -ne 0) {
            if ($PassNumber -lt $RetryIncompletePasses) {
                Write-Warning "$($Batch.Name) validator process exited $JavaExit; retrying after 10 seconds"
                Start-Sleep -Seconds 10
                continue
            }
            throw "JForex validator process failed for $($Batch.Name) with exit code $JavaExit"
        }

        & $Python src\events\resume_jforex_validation.py merge `
            --input $InputFile --result-dir $ResultDir `
            --attempt-result $AttemptResult --plan-file $PlanFile
        if ($LASTEXITCODE -ne 0) { throw "Result merge failed for $($Batch.Name)" }

        $Rows = Test-BatchOutputs -InputFile $InputFile -ResultDir $ResultDir -BatchName $Batch.Name
        $Pass = @($Rows | Where-Object { $_.status -eq 'PASS' }).Count
        $Incomplete = @($Rows | Where-Object { $_.status -eq 'INCOMPLETE' }).Count
        $Mismatch = @($Rows | Where-Object { $_.status -eq 'MISMATCH' }).Count
        $Errors = @($Rows | Where-Object { $_.status -eq 'ERROR' }).Count
        Write-Host ("{0}: PASS={1} INCOMPLETE={2} MISMATCH={3} ERROR={4}" -f `
            $Batch.Name, $Pass, $Incomplete, $Mismatch, $Errors)
        if ($Incomplete -eq 0) { break }
    }
}

if ($BatchSize -lt 1) { throw 'BatchSize must be at least 1' }
if ($RetryIncompletePasses -lt 0) { throw 'RetryIncompletePasses cannot be negative' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing venv Python: $Python" }
if (-not (Test-Path -LiteralPath $EventParquet)) { throw "Missing validated event table: $EventParquet" }
if (-not $env:DUKASCOPY_USER -or -not $env:DUKASCOPY_PASSWORD) {
    throw 'DUKASCOPY_USER and DUKASCOPY_PASSWORD must be loaded in this PowerShell process'
}

Push-Location $Root
try {
    Write-Host '=== MarketFusion full Dukascopy event validation ==='
    Write-Host 'Mode: read-only; no trading/order API is used by the validator.'

    Write-Host '[1/8] Complete non-credentialed preflight'
    & (Join-Path $PSScriptRoot 'preflight.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Repository preflight failed' }

    Write-Host '[2/8] Validate canonical event table'
    & $Python src\events\validate_event_table.py
    if ($LASTEXITCODE -ne 0) { throw 'Canonical event-table validation failed' }

    Write-Host '[3/8] Export ALL model-eligible events'
    & $Python src\events\export_events_for_jforex.py --output $AllEvents
    if ($LASTEXITCODE -ne 0) { throw 'Full JForex event export failed' }

    Write-Host '[4/8] Prepare deterministic batches without discarding compatible results'
    & $Python src\events\prepare_jforex_validation_batches.py `
        --input $AllEvents --output-dir $BatchRoot --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw 'Batch preparation failed' }

    Write-Host '[5/8] Build isolated Java runtime classpath'
    $ClasspathFile = Join-Path $JForexRoot 'target\runtime-classpath.txt'
    & mvn -B -ntp -f (Join-Path $JForexRoot 'pom.xml') dependency:build-classpath `
        "-Dmdep.outputFile=$ClasspathFile"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $ClasspathFile)) {
        throw 'Failed to build JForex runtime classpath'
    }
    $DependencyClasspath = (Get-Content -LiteralPath $ClasspathFile -Raw).Trim()
    $RuntimeClasspath = "$(Join-Path $JForexRoot 'target\classes');$DependencyClasspath"
    $JavaExecutable = Join-Path $env:JAVA_HOME 'bin\java.exe'
    if (-not (Test-Path -LiteralPath $JavaExecutable)) { throw "Missing Java executable: $JavaExecutable" }

    Write-Host '[6/8] Run/reuse batches; retry only compatible INCOMPLETE events'
    $BatchDirs = @(Get-ChildItem -Path $BatchRoot -Directory -Filter 'batch_*' | Sort-Object Name)
    if ($BatchDirs.Count -eq 0) { throw 'No validation batches were created' }
    for ($Index = 0; $Index -lt $BatchDirs.Count; $Index++) {
        Invoke-ValidatorAttempt -Batch $BatchDirs[$Index] -RuntimeClasspath $RuntimeClasspath `
            -JavaExecutable $JavaExecutable -Ordinal ($Index + 1) -Total $BatchDirs.Count
    }

    Write-Host '[7/8] Aggregate with event-set, metadata, numeric, and status integrity gates'
    & $Python src\events\aggregate_jforex_validation.py `
        --expected-events $AllEvents --batch-root $BatchRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Full validation contains MISMATCH/ERROR or failed an integrity check'
    }

    Write-Host '[8/8] Complete'
    Write-Host '=== FULL VALIDATION RUN COMPLETE ==='
    Write-Host 'INCOMPLETE is quarantined provider unavailability; it is never converted to PASS.'
} finally {
    Pop-Location
}
