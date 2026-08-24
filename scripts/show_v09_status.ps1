$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
python -c "from src.research.v09_runner import show_status; show_status()"
