param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$atlasRoot = $PSScriptRoot
$atlasPython = Join-Path $atlasRoot '.tools/attention-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $atlasPython)) {
    throw 'Attention environment missing. Follow docs/attention-lab.md to install the pinned worker environment.'
}
$atlasUrl = "http://127.0.0.1:$Port/"
$atlasRunning = $false
$atlasOwnerText = & $atlasPython (Join-Path $atlasRoot 'benchmarks/atlas_service_owner.py') $atlasRoot $Port
if ($LASTEXITCODE -ne 0) { throw 'The selected port could not be verified as this Atlas checkout.' }
$atlasOwner = $atlasOwnerText | ConvertFrom-Json
if ($atlasOwner.found -and $atlasOwner.entrypoint -eq 'lab.py') {
    & $atlasPython (Join-Path $atlasRoot 'benchmarks/atlas_service_owner.py') $atlasRoot $Port --stop --require-idle
    if ($LASTEXITCODE -ne 0) { throw 'The older Atlas service could not be upgraded while a job is active.' }
}
try {
    $atlasSession = Invoke-RestMethod "${atlasUrl}api/session" -TimeoutSec 2
    $atlasRunning = [bool]$atlasSession.token -and [bool]$atlasSession.runtime_tuning
} catch { }
if (-not $atlasRunning) {
    & $atlasPython (Join-Path $atlasRoot 'benchmarks/prepare_attention_history.py')
    if ($LASTEXITCODE -ne 0) { throw 'Reference history preparation failed; original evidence was not changed.' }
    $atlasLogs = Join-Path $atlasRoot 'artifacts/attention-lab-server'
    New-Item -ItemType Directory -Force -Path $atlasLogs | Out-Null
    $atlasProcess = Start-Process -FilePath $atlasPython -ArgumentList @('tune.py', 'serve', '--port', "$Port") -WorkingDirectory $atlasRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $atlasLogs 'stdout.log') -RedirectStandardError (Join-Path $atlasLogs 'stderr.log')
    for ($atlasAttempt = 0; $atlasAttempt -lt 60; $atlasAttempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $atlasSession = Invoke-RestMethod "${atlasUrl}api/session" -TimeoutSec 1
            if ($atlasSession.token -and $atlasSession.runtime_tuning) { $atlasRunning = $true; break }
        } catch { }
        if ($atlasProcess.HasExited) { break }
    }
    if (-not $atlasRunning) { throw "Atlas could not start. Inspect $atlasLogs" }
}
Write-Output "Atlas Attention Lab: $atlasUrl"
if (-not $NoBrowser) { Start-Process $atlasUrl }
