param(
    [switch]$RunJForexSample
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

Write-Host '=== MarketFusion AI preflight ==='

$Python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Missing virtual-environment Python: $Python"
}

Write-Host '[1/6] Python syntax compile'
& $Python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { throw 'Python compileall failed' }

Write-Host '[2/6] Repository safety policy'
& $Python scripts\repo_safety_check.py
if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }

Write-Host '[3/6] Unit tests'
& $Python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Python unit tests failed' }

Write-Host '[4/6] Event-table validation when local data exists'
$EventParquet = Join-Path $Root 'data\events\bls_release_events.parquet'
if (Test-Path $EventParquet) {
    & $Python src\events\validate_event_table.py
    if ($LASTEXITCODE -ne 0) { throw 'Event-table validation failed' }
} else {
    Write-Host 'SKIP: local event parquet is not present'
}

Write-Host '[5/6] Java/Maven environment and JForex compile'
if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
    $MavenHome = 'C:\tools\apache-maven-3.9.16-bin\apache-maven-3.9.16'
    if (-not (Test-Path (Join-Path $MavenHome 'bin\mvn.cmd'))) {
        throw 'Maven is not on PATH and the known Maven installation was not found'
    }
    $env:MAVEN_HOME = $MavenHome
    $env:Path = "$env:MAVEN_HOME\bin;$env:Path"
}

$OracleJava8 = 'C:\Program Files\Java\jdk1.8.0_501'
if (Test-Path (Join-Path $OracleJava8 'bin\java.exe')) {
    $env:JAVA_HOME = $OracleJava8
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
}

java -version
mvn -version
Push-Location (Join-Path $Root 'jforex-event-exporter')
try {
    mvn clean compile
    if ($LASTEXITCODE -ne 0) { throw 'JForex Maven compile failed' }
} finally {
    Pop-Location
}

Write-Host '[6/6] Optional live JForex validation sample'
if ($RunJForexSample) {
    if (-not $env:DUKASCOPY_USER -or -not $env:DUKASCOPY_PASSWORD) {
        throw 'RunJForexSample requires DUKASCOPY_USER and DUKASCOPY_PASSWORD in this process environment'
    }
    if (-not (Test-Path $EventParquet)) {
        throw 'RunJForexSample requires data\events\bls_release_events.parquet'
    }

    $Sample = Join-Path $Root 'data\dukascopy\tick_validation_sample.tsv'
    & $Python src\events\export_events_for_jforex.py --sample-per-year 1 --output $Sample
    if ($LASTEXITCODE -ne 0) { throw 'JForex validation-sample export failed' }

    Push-Location (Join-Path $Root 'jforex-event-exporter')
    try {
        mvn exec:java `
          '-Dexec.mainClass=ai.marketfusion.jforex.TickValidationBatchRobust' `
          '-Dexec.args=..\data\dukascopy\tick_validation_sample.tsv ..\data\dukascopy\tick_validation_robust'
        if ($LASTEXITCODE -ne 0) { throw 'Robust JForex validation process failed' }
    } finally {
        Pop-Location
    }
} else {
    Write-Host 'SKIP: use -RunJForexSample to run the credentialed Dukascopy sample'
}

Write-Host '=== PREFLIGHT COMPLETE ==='
