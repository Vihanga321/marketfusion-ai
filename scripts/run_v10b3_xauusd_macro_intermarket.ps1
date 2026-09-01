param(
    [switch]$NoPersist,
    [switch]$Offline,
    [switch]$BoardOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Offline -and $BoardOnly) {
    throw '-Offline and -BoardOnly are mutually exclusive'
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtual-environment Python: $Python"
}

$HadLokyCpuCount = Test-Path Env:\LOKY_MAX_CPU_COUNT
$PreviousLokyCpuCount = $env:LOKY_MAX_CPU_COUNT
if (-not $HadLokyCpuCount) {
    $LogicalCores = [Math]::Max(1, [Environment]::ProcessorCount)
    $PhysicalCores = 0
    try {
        $CpuSummary = Get-CimInstance Win32_Processor -ErrorAction Stop |
            Measure-Object -Property NumberOfCores -Sum
        if ($null -ne $CpuSummary.Sum) { $PhysicalCores = [int]$CpuSummary.Sum }
    } catch { $PhysicalCores = 0 }
    if ($PhysicalCores -gt 0 -and $PhysicalCores -le $LogicalCores) {
        $WorkerCeiling = $PhysicalCores
        $WorkerCeilingSource = 'Windows CIM physical cores'
    } elseif ($LogicalCores -gt 1) {
        $WorkerCeiling = $LogicalCores - 1
        $WorkerCeilingSource = 'conservative logical-core fallback'
    } else {
        $WorkerCeiling = 1
        $WorkerCeilingSource = 'single-core fallback'
    }
    $env:LOKY_MAX_CPU_COUNT = [string]$WorkerCeiling
} else {
    $WorkerCeilingSource = 'existing LOKY_MAX_CPU_COUNT environment override'
}

Push-Location $Root
try {
    & $Python scripts\repo_safety_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository safety validation failed' }

    Write-Host 'MarketFusion V1.0B.3 XAUUSD verified macro/intermarket evidence research'
    if ($BoardOnly) {
        Write-Host 'Provider mode: OFFICIAL FEDERAL RESERVE BOARD ONLY (full-history DDP CSV).'
    } elseif ($Offline) {
        Write-Host 'Provider mode: OFFLINE VERIFIED LOCAL CONTEXT STORE.'
    } else {
        Write-Host 'Preferred provider: Federal Reserve/FRED. Official Board fallback is available if FRED is unreachable.'
    }
    Write-Host 'No fabricated news, sentiment, fundamentals, or market values are used.'
    Write-Host 'Current-vintage history is RESEARCH ONLY and cannot promote a production champion.'
    Write-Host 'The previously exposed final XAUUSD tail remains quarantined.'
    Write-Host 'Automatic execution remains DISABLED; runtime remains SHADOW_ADVISORY_ONLY.'
    Write-Host "Research CPU worker ceiling: $env:LOKY_MAX_CPU_COUNT ($WorkerCeilingSource)"

    if ($BoardOnly) {
        & $Python -m src.intelligence.gold_macro_fed_board_fallback
        $FallbackExit = $LASTEXITCODE
        if ($FallbackExit -ne 0) {
            throw 'Official Federal Reserve Board context bootstrap failed'
        }
        Write-Host 'Official Federal Reserve Board context: PASS'
        Write-Host 'Running V1.0B.3 from the verified local context store...'
        $OfflineArguments = @('-m', 'src.research.v10b3_xauusd_macro_intermarket', '--offline')
        if ($NoPersist) { $OfflineArguments += '--no-persist' }
        & $Python @OfflineArguments
        $ResearchExit = $LASTEXITCODE
    } else {
        $Arguments = @('-m', 'src.research.v10b3_xauusd_macro_intermarket')
        if ($NoPersist) { $Arguments += '--no-persist' }
        if ($Offline) { $Arguments += '--offline' }

        & $Python @Arguments
        $ResearchExit = $LASTEXITCODE

        if ($ResearchExit -ne 0 -and -not $Offline) {
            Write-Warning 'FRED refresh/research did not complete. Trying official federalreserve.gov fallback context.'
            & $Python -m src.intelligence.gold_macro_fed_board_fallback
            $FallbackExit = $LASTEXITCODE
            if ($FallbackExit -eq 0) {
                Write-Host 'Official Federal Reserve Board fallback context: PASS'
                Write-Host 'Re-running V1.0B.3 from the verified local context store...'
                $OfflineArguments = @('-m', 'src.research.v10b3_xauusd_macro_intermarket', '--offline')
                if ($NoPersist) { $OfflineArguments += '--no-persist' }
                & $Python @OfflineArguments
                $ResearchExit = $LASTEXITCODE
            } else {
                $ResearchExit = $FallbackExit
            }
        }
    }

    if ($ResearchExit -ne 0) { throw 'V1.0B.3 XAUUSD macro/intermarket research failed' }
} finally {
    Pop-Location
    if ($HadLokyCpuCount) { $env:LOKY_MAX_CPU_COUNT = $PreviousLokyCpuCount }
    else { Remove-Item Env:\LOKY_MAX_CPU_COUNT -ErrorAction SilentlyContinue }
}
