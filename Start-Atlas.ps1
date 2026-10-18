param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$atlasRoot = $PSScriptRoot
$atlasPython = Join-Path $atlasRoot '.tools/attention-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $atlasPython)) {
    throw 'Attention environment missing. Follow docs/attention-lab.md to install the pinned worker environment.'
}
$atlasUrl = "http://127.0.0.1:$Port/"
$atlasRunning = $false
try {
    $atlasSession = Invoke-RestMethod "${atlasUrl}api/session" -TimeoutSec 2
    $atlasRunning = [bool]$atlasSession.token
} catch { }
if (-not $atlasRunning) {
    $atlasLogs = Join-Path $atlasRoot 'artifacts/attention-lab-server'
    New-Item -ItemType Directory -Force -Path $atlasLogs | Out-Null
    $atlasProcess = Start-Process -FilePath $atlasPython -ArgumentList @('lab.py', 'serve', '--port', "$Port") -WorkingDirectory $atlasRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $atlasLogs 'stdout.log') -RedirectStandardError (Join-Path $atlasLogs 'stderr.log')
    for ($atlasAttempt = 0; $atlasAttempt -lt 60; $atlasAttempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $atlasSession = Invoke-RestMethod "${atlasUrl}api/session" -TimeoutSec 1
            if ($atlasSession.token) { $atlasRunning = $true; break }
        } catch { }
        if ($atlasProcess.HasExited) { break }
    }
    if (-not $atlasRunning) { throw "Atlas could not start. Inspect $atlasLogs" }
}
Write-Output "Atlas Attention Lab: $atlasUrl"
if (-not $NoBrowser) { Start-Process $atlasUrl }
