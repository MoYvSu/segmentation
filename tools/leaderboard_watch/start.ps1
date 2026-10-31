param([switch]$NoOpen)
$ErrorActionPreference = 'Stop'
$watchRoot = Split-Path -Parent $PSScriptRoot
$watchRoot = Split-Path -Parent $watchRoot
$watchData = Join-Path $watchRoot 'output\leaderboard_watch'
$watchUrl = 'http://127.0.0.1:8876/'
$watchReady = $false
try {
    $watchHealth = Invoke-RestMethod -Uri ($watchUrl + 'api/health') -TimeoutSec 2
    if ($watchHealth.app -eq 'aicomp-leaderboard-watch') { $watchReady = $true }
    else { throw 'Port 8876 belongs to another application.' }
} catch {
    if ($_.Exception.Message -eq 'Port 8876 belongs to another application.') { throw }
}
if (-not $watchReady) {
    New-Item -ItemType Directory -Force -Path $watchData | Out-Null
    $watchPython = @(Get-Command python -CommandType Application -ErrorAction Stop)[0].Source
    $watchApp = Join-Path $PSScriptRoot 'app.py'
    $watchArguments = @(('"' + $watchApp + '"'), 'serve', '--interval', '300')
    $watchProcess = Start-Process -FilePath $watchPython -ArgumentList $watchArguments -WorkingDirectory $watchRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $watchData 'stdout.log') -RedirectStandardError (Join-Path $watchData 'stderr.log') -PassThru
    for ($watchTry = 0; $watchTry -lt 24; $watchTry++) {
        Start-Sleep -Milliseconds 250
        try {
            $watchHealth = Invoke-RestMethod -Uri ($watchUrl + 'api/health') -TimeoutSec 1
            if ($watchHealth.app -eq 'aicomp-leaderboard-watch') { $watchReady = $true; break }
        } catch { }
        if ($watchProcess.HasExited) { break }
    }
    if (-not $watchReady) { throw ('Startup failed. Check ' + (Join-Path $watchData 'stderr.log')) }
}
Write-Output ('Leaderboard monitor: ' + $watchUrl)
if (-not $NoOpen) { Start-Process -FilePath $watchUrl }
