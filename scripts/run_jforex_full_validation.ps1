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

Set-Location $Root
Write-Host '=== MarketFusion full Dukascopy event validation ==='
Write-Host 'Mode: read-only; no trading/order API is used by the validator.'

if ($BatchSize -lt 1) {
    throw 'BatchSize must be at least 1'
}
if ($RetryIncompletePasses -lt 0) {
    throw 'RetryIncompletePasses cannot be negative'
}
if (-not (Test-Path $Python)) {
    throw "Missing venv Python: $Python"
}
if (-not (Test-Path $EventParquet)) {
    throw "Missing validated event table: $EventParquet"
}
if (-not $env:DUKASCOPY_USER -or -not $env:DUKASCOPY_PASSWORD) {
    throw 'DUKASCOPY_USER and DUKASCOPY_PASSWORD must be loaded in this PowerShell process'
}

# Restore the known working local toolchain when a fresh VS Code terminal does
# not inherit Maven/Java configuration.
$OracleJava8 = 'C:\Program Files\Java\jdk1.8.0_501'
if (Test-Path (Join-Path $OracleJava8 'bin\java.exe')) {
    $env:JAVA_HOME = $OracleJava8
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
}
if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
    $MavenHome = 'C:\tools\apache-maven-3.9.16-bin\apache-maven-3.9.16'
    if (-not (Test-Path (Join-Path $MavenHome 'bin\mvn.cmd'))) {
        throw 'Maven is not on PATH and the known Maven installation was not found'
    }
    $env:MAVEN_HOME = $MavenHome
    $env:Path = "$env:MAVEN_HOME\bin;$env:Path"
}

function Get-TsvRows {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing TSV: $Path"
    }
    return @(Import-Csv -Path $Path -Delimiter "`t")
}

function Test-BatchOutputs {
    param(
        [Parameter(Mandatory = $true)][string]$InputFile,
        [Parameter(Mandatory = $true)][string]$ResultDir,
        [Parameter(Mandatory = $true)][string]$BatchName
    )

    $Summary = Join-Path $ResultDir 'robust_tick_validation_summary.tsv'
    $Chunks = Join-Path $ResultDir 'robust_tick_chunk_status.tsv'
    if (-not (Test-Path $Summary) -or -not (Test-Path $Chunks)) {
        throw "Validator outputs are missing for $BatchName"
    }

    $ExpectedRows = Get-TsvRows -Path $InputFile
    $SummaryRows = Get-TsvRows -Path $Summary
    if ($SummaryRows.Count -ne $ExpectedRows.Count) {
        throw "Batch output row-count mismatch for $BatchName: expected $($ExpectedRows.Count), got $($SummaryRows.Count)"
    }

    $ExpectedIds = @($ExpectedRows | ForEach-Object { $_.event_id } | Sort-Object)
    $ActualIds = @($SummaryRows | ForEach-Object { $_.event_id } | Sort-Object)
    if (($ExpectedIds -join "`n") -ne ($ActualIds -join "`n")) {
        throw "Batch output event-set mismatch for $BatchName"
    }

    $BadStatuses = @($SummaryRows | Where-Object { $_.status -notin @('PASS', 'INCOMPLETE', 'MISMATCH', 'ERROR') })
    if ($BadStatuses.Count -gt 0) {
        throw "Unknown validation status in $BatchName"
    }

    return $SummaryRows
}

function Invoke-JForexBatch {
    param(
        [Parameter(Mandatory = $true)][System.IO.DirectoryInfo]$Batch,
        [Parameter(Mandatory = $true)][int]$Ordinal,
        [Parameter(Mandatory = $true)][int]$Total
    )

    $InputFile = Join-Path $Batch.FullName 'events.tsv'
    $ResultDir = Join-Path $Batch.FullName 'result'
    $MaxAttempts = 1 + $RetryIncompletePasses

    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
        if (Test-Path $ResultDir) {
            Remove-Item -Recurse -Force $ResultDir
        }
        New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

        Write-Host ("--- {0} ({1}/{2}) attempt {3}/{4} ---" -f $Batch.Name, $Ordinal, $Total, $Attempt, $MaxAttempts)
        Push-Location $JForexRoot
        try {
            # RobustV2 explicitly exits the standalone Maven JVM after JForex
            # disconnects. This prevents SDK background threads from hanging the
            # PowerShell batch loop after a strategy has already stopped cleanly.
            mvn exec:java `
                ("-Dexec.mainClass={0}" -f $ValidatorClass) `
                ("-Dexec.args={0} {1}" -f $InputFile, $ResultDir)
            $MavenExit = $LASTEXITCODE
        } finally {
            Pop-Location
        }

        if ($MavenExit -ne 0) {
            if ($Attempt -lt $MaxAttempts) {
                Write-Warning "$($Batch.Name) validator process exited $MavenExit; retrying after 10 seconds"
                Start-Sleep -Seconds 10
                continue
            }
            throw "JForex validator process failed for $($Batch.Name) with exit code $MavenExit"
        }

        $Rows = Test-BatchOutputs -InputFile $InputFile -ResultDir $ResultDir -BatchName $Batch.Name
        $Mismatch = @($Rows | Where-Object { $_.status -eq 'MISMATCH' }).Count
        $Errors = @($Rows | Where-Object { $_.status -eq 'ERROR' }).Count
        $Incomplete = @($Rows | Where-Object { $_.status -eq 'INCOMPLETE' }).Count
        $Pass = @($Rows | Where-Object { $_.status -eq 'PASS' }).Count

        Write-Host ("{0}: PASS={1} INCOMPLETE={2} MISMATCH={3} ERROR={4}" -f $Batch.Name, $Pass, $Incomplete, $Mismatch, $Errors)

        if (($Mismatch -gt 0) -or ($Errors -gt 0)) {
            # Preserve the batch result and continue. The final aggregator will
            # report every hard failure across the complete event set instead of
            # stopping at the first bad event.
            return
        }

        if (($Incomplete -gt 0) -and ($Attempt -lt $MaxAttempts)) {
            Write-Warning "$($Batch.Name) has $Incomplete provider-incomplete event(s); retrying the whole batch once"
            Start-Sleep -Seconds 10
            continue
        }

        return
    }
}

Write-Host '[1/7] Repository preflight'
& $Python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety preflight failed' }

Write-Host '[2/7] Validate canonical event table'
& $Python src\events\validate_event_table.py
if ($LASTEXITCODE -ne 0) { throw 'Canonical event-table validation failed' }

Write-Host '[3/7] Export ALL model-eligible events'
& $Python src\events\export_events_for_jforex.py --output $AllEvents
if ($LASTEXITCODE -ne 0) { throw 'Full JForex event export failed' }

Write-Host '[4/7] Prepare deterministic validation batches'
& $Python src\events\prepare_jforex_validation_batches.py `
    --input $AllEvents `
    --output-dir $BatchRoot `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'Batch preparation failed' }

Write-Host '[5/7] Compile JForex validator with Java 8'
java -version
mvn -version
Push-Location $JForexRoot
try {
    mvn clean compile
    if ($LASTEXITCODE -ne 0) { throw 'JForex Maven compile failed' }
} finally {
    Pop-Location
}

Write-Host '[6/7] Run every batch against Dukascopy historical data'
$BatchDirs = @(Get-ChildItem -Path $BatchRoot -Directory -Filter 'batch_*' | Sort-Object Name)
if ($BatchDirs.Count -eq 0) {
    throw 'No validation batches were created'
}

for ($Index = 0; $Index -lt $BatchDirs.Count; $Index++) {
    Invoke-JForexBatch -Batch $BatchDirs[$Index] -Ordinal ($Index + 1) -Total $BatchDirs.Count
}

Write-Host '[7/7] Aggregate and integrity-check all event results'
& $Python src\events\aggregate_jforex_validation.py `
    --expected-events $AllEvents `
    --batch-root $BatchRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Full validation contains a reconstruction MISMATCH/ERROR or failed integrity checks'
}

Write-Host '=== FULL VALIDATION RUN COMPLETE ==='
Write-Host 'Read reports\dukascopy_full_validation_summary.txt for PASS/INCOMPLETE counts.'
Write-Host 'INCOMPLETE means provider data were unavailable; it is never silently filled.'
