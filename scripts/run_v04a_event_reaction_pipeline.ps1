param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$JForexRoot = Join-Path $Root 'jforex-event-exporter'
$EligibleInput = Join-Path $Root 'data\dukascopy\v04a_reaction\eligible_events.tsv'
$ReactionOutput = Join-Path $Root 'data\dukascopy\v04a_reaction'

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Missing virtual-environment Python: $Python"
    }

    Write-Host '[1/7] Verify immutable production and forensic evidence'
    & $Python src\events\verify_v04a_evidence.py
    if ($LASTEXITCODE -ne 0) { throw 'Frozen V0.4A evidence verification failed' }

    Write-Host '[2/7] Build final adjudication and exact strict-PASS manifest'
    & $Python src\events\build_v04a_adjudication.py
    if ($LASTEXITCODE -ne 0) { throw 'V0.4A adjudication or eligible-manifest build failed' }

    Write-Host '[3/7] Run all non-live quality and safety checks'
    & (Join-Path $PSScriptRoot 'preflight.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; reaction collection was not started' }

    $CredentialsAvailable = [bool]$env:DUKASCOPY_USER -and [bool]$env:DUKASCOPY_PASSWORD
    if (-not $CredentialsAvailable) {
        Write-Host '[4/7] REACTION_EXTRACTION_NOT_RUN: process-scoped Dukascopy credentials unavailable'
        Write-Host '[5/7] Skip JForex classpath build'
        Write-Host '[6/7] Build honest empty/not-run reaction artifacts'
        & $Python src\events\build_v04a_event_reactions.py
        if ($LASTEXITCODE -ne 0) { throw 'Reaction not-run reporting failed' }
        Write-Host '[7/7] Reverify immutable evidence'
        & $Python src\events\verify_v04a_evidence.py
        if ($LASTEXITCODE -ne 0) { throw 'Frozen evidence changed during the pipeline' }
        Write-Host 'REACTION_EXTRACTION_NOT_RUN'
        return
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

    Write-Host '[4/7] Build read-only JForex collector classpath'
    $ClasspathFile = Join-Path $JForexRoot 'target\runtime-classpath.txt'
    & mvn -B -ntp -f (Join-Path $JForexRoot 'pom.xml') dependency:build-classpath `
        "-Dmdep.outputFile=$ClasspathFile"
    if ($LASTEXITCODE -ne 0) { throw 'JForex runtime classpath build failed' }
    $Dependencies = (Get-Content -LiteralPath $ClasspathFile -Raw).Trim()
    $RuntimeClasspath = "$(Join-Path $JForexRoot 'target\classes');$Dependencies"

    Write-Host '[5/7] Retrieve or fingerprint-verify only 242 strict-PASS windows'
    & (Join-Path $env:JAVA_HOME 'bin\java.exe') -cp $RuntimeClasspath `
        'ai.marketfusion.jforex.EventReactionCollector' $EligibleInput $ReactionOutput $Root
    if ($LASTEXITCODE -ne 0) { throw 'Read-only reaction collection failed' }

    Write-Host '[6/7] Build and validate reaction outcomes'
    & $Python src\events\build_v04a_event_reactions.py
    if ($LASTEXITCODE -ne 0) { throw 'Reaction dataset build failed' }

    Write-Host '[7/7] Reverify immutable evidence'
    & $Python src\events\verify_v04a_evidence.py
    if ($LASTEXITCODE -ne 0) { throw 'Frozen evidence changed during the pipeline' }
} finally {
    Pop-Location
}
