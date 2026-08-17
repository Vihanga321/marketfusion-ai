param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$JForexRoot = Join-Path $Root 'jforex-event-exporter'
$Input = Join-Path $Root 'data\dukascopy\mismatch_diagnostic\events.tsv'
$Minutes = Join-Path $Root 'reports\dukascopy_mismatch_minutes.tsv'
$Context = Join-Path $Root 'reports\dukascopy_mismatch_tick_context.tsv'
$Report = Join-Path $Root 'reports\dukascopy_mismatch_diagnostic.txt'

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }

    Write-Host '[1/4] Verify production mismatch set and prepare exactly four events'
    & $Python src\events\prepare_dukascopy_mismatch_diagnostic.py --output $Input
    if ($LASTEXITCODE -ne 0) { throw 'Four-event input preparation failed' }

    Write-Host '[2/4] Run all non-live checks before credentialed history access'
    & (Join-Path $PSScriptRoot 'preflight.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; diagnostic was not started' }

    if (-not $env:DUKASCOPY_USER -or -not $env:DUKASCOPY_PASSWORD) {
        throw 'Targeted diagnostic requires DUKASCOPY_USER and DUKASCOPY_PASSWORD in this process environment'
    }

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

    Write-Host '[3/4] Build the read-only diagnostic runtime classpath'
    $ClasspathFile = Join-Path $JForexRoot 'target\runtime-classpath.txt'
    & mvn -B -ntp -f (Join-Path $JForexRoot 'pom.xml') dependency:build-classpath `
        "-Dmdep.outputFile=$ClasspathFile"
    if ($LASTEXITCODE -ne 0) { throw 'JForex runtime classpath build failed' }
    $Dependencies = (Get-Content -LiteralPath $ClasspathFile -Raw).Trim()
    $RuntimeClasspath = "$(Join-Path $JForexRoot 'target\classes');$Dependencies"

    Write-Host '[4/4] Retrieve and diagnose exactly four approved events'
    & (Join-Path $env:JAVA_HOME 'bin\java.exe') -cp $RuntimeClasspath `
        'ai.marketfusion.jforex.TickMismatchDiagnostic' $Input $Minutes $Context $Report
    if ($LASTEXITCODE -ne 0) { throw 'Targeted Dukascopy mismatch diagnostic failed' }

    Write-Host "Minute evidence: $Minutes"
    Write-Host "Tick context:   $Context"
    Write-Host "Report:         $Report"
} finally {
    Pop-Location
}
