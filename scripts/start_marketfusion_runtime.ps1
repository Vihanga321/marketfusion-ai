param(
    [int]$RuntimeIntervalSeconds = 15,
    [int]$MarketPollSeconds = 15,
    [int]$IntelligencePollSeconds = 300
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root 'venv\Scripts\python.exe'
$PidRoot = Join-Path $Root 'data\runtime\v06\pids'

function Start-MarketFusionWorker {
    param([string]$Name, [string[]]$Arguments)
    $PidFile = Join-Path $PidRoot "$Name.pid"
    if (Test-Path -LiteralPath $PidFile) {
        $ExistingPid = 0
        if ([int]::TryParse((Get-Content -Raw -LiteralPath $PidFile).Trim(), [ref]$ExistingPid)) {
            $Existing = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
            if ($null -ne $Existing -and $Existing.ProcessName -like 'python*') {
                Write-Host "$Name already running as PID $ExistingPid; not starting a duplicate."
                return
            }
        }
        Remove-Item -LiteralPath $PidFile -Force
    }
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
    Write-Host "Started $Name as PID $($Process.Id)."
}

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing virtual-environment Python: $Python" }
    New-Item -ItemType Directory -Force -Path $PidRoot | Out-Null
    Start-MarketFusionWorker 'v05a_collector' @('-m', 'src.marketdata.v05a_runner', '--poll-seconds', "$MarketPollSeconds", '--overlap-bars', '8')
    Start-MarketFusionWorker 'v05b_collector' @('-m', 'src.intelligence.v05b_runner', '--interval', "$IntelligencePollSeconds")
    Write-Host 'V0.5C artifacts and V0.5D/V0.4D event state are consumed read-only; training is not started.'
    & $Python -m src.runtime.v06c_runner --continuous --interval-seconds $RuntimeIntervalSeconds
    if ($LASTEXITCODE -ne 0) { throw 'MarketFusion unified runtime stopped with an error' }
} finally { Pop-Location }
