param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$atlasPython = Join-Path $PSScriptRoot '.tools/attention-env/Scripts/python.exe'
& $atlasPython (Join-Path $PSScriptRoot 'benchmarks/atlas_service_owner.py') $PSScriptRoot $Port --stop
if ($LASTEXITCODE -ne 0) { throw 'Atlas was not stopped; inspect the diagnostic above.' }
