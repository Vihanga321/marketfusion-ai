param([switch]$RunJForexSample)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$EventParquet = Join-Path $Root 'data\events\bls_release_events.parquet'
$JForexRoot = Join-Path $Root 'jforex-event-exporter'

Push-Location $Root
try {
    Write-Host '=== MarketFusion AI preflight ==='
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }

    Write-Host '[1/7] Python syntax compile'
    & $Python -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Python compileall failed' }

    Write-Host '[2/7] Repository safety policy'
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety check failed' }

    Write-Host '[3/7] Python unit tests'
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Python unit tests failed' }

    Write-Host '[4/7] PowerShell parser validation'
    $PowerShellFiles = @(Get-ChildItem -Path (Join-Path $Root 'scripts') -Filter '*.ps1' -File)
    foreach ($File in $PowerShellFiles) {
        $Tokens = $null
        $Errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $File.FullName, [ref]$Tokens, [ref]$Errors
        ) | Out-Null
        if ($Errors.Count -gt 0) {
            $Errors | ForEach-Object { Write-Error $_.Message }
            throw "PowerShell syntax validation failed: $($File.Name)"
        }
    }
    Write-Host "PowerShell syntax PASS: $($PowerShellFiles.Count) file(s)"

    Write-Host '[5/7] Event-table validation when local data exists'
    if (Test-Path -LiteralPath $EventParquet) {
        & $Python src\events\validate_event_table.py
        if ($LASTEXITCODE -ne 0) { throw 'Event-table validation failed' }
    } else {
        Write-Host 'SKIP: local event parquet is not present'
    }

    Write-Host '[6/7] Java 8/Maven compile and reconstruction unit tests'
    $MavenHome = 'C:\tools\apache-maven-3.9.16-bin\apache-maven-3.9.16'
    if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
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
    & mvn -B -ntp -f (Join-Path $JForexRoot 'pom.xml') clean test
    if ($LASTEXITCODE -ne 0) { throw 'JForex Maven tests failed' }

    Write-Host '[7/7] Optional credentialed JForex validation sample'
    if ($RunJForexSample) {
        if (-not $env:DUKASCOPY_USER -or -not $env:DUKASCOPY_PASSWORD) {
            throw 'RunJForexSample requires DUKASCOPY_USER and DUKASCOPY_PASSWORD in this process environment'
        }
        if (-not (Test-Path -LiteralPath $EventParquet)) {
            throw 'RunJForexSample requires data\events\bls_release_events.parquet'
        }
        $Sample = Join-Path $Root 'data\dukascopy\tick_validation_sample.tsv'
        $SampleOutput = Join-Path $Root 'data\dukascopy\tick_validation_robust_v2'
        & $Python src\events\export_events_for_jforex.py --sample-per-year 1 --output $Sample
        if ($LASTEXITCODE -ne 0) { throw 'JForex validation-sample export failed' }
        $ClasspathFile = Join-Path $JForexRoot 'target\runtime-classpath.txt'
        & mvn -B -ntp -f (Join-Path $JForexRoot 'pom.xml') dependency:build-classpath `
            "-Dmdep.outputFile=$ClasspathFile"
        if ($LASTEXITCODE -ne 0) { throw 'JForex runtime classpath build failed' }
        $Dependencies = (Get-Content -LiteralPath $ClasspathFile -Raw).Trim()
        $RuntimeClasspath = "$(Join-Path $JForexRoot 'target\classes');$Dependencies"
        & (Join-Path $env:JAVA_HOME 'bin\java.exe') -cp $RuntimeClasspath `
            'ai.marketfusion.jforex.TickValidationBatchRobustV2' $Sample $SampleOutput
        if ($LASTEXITCODE -ne 0) { throw 'Robust JForex validation process failed' }
    } else {
        Write-Host 'SKIP: use -RunJForexSample to run the credentialed Dukascopy sample'
    }

    Write-Host '=== PREFLIGHT COMPLETE ==='
} finally {
    Pop-Location
}
