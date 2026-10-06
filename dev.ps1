<#
.SYNOPSIS
  Start or stop the lab's backend and frontend, clearing stale ports first.

.DESCRIPTION
  Repeated "Port 5173 is already in use" and "WinError 10048" failures come from
  an earlier run that was closed without releasing its listener: the port stays
  bound by an orphan, and the next start dies after a full model warm-up. This
  script kills whatever holds the two ports before starting, so a stale process
  can never block a fresh start.

  Start both detached and return immediately, rather than holding a foreground
  process: Ctrl+C on one service would otherwise leave the other orphaned, which
  is the problem this script exists to remove.

.PARAMETER Stop
  Kill anything holding the ports and exit without starting anything.

.PARAMETER Logs
  Follow the backend and frontend logs instead of starting anything.

.EXAMPLE
  .\dev.ps1
  .\dev.ps1 -Stop
  .\dev.ps1 -Logs
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$Logs
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendPort = 8000
$FrontendPort = 5173
$LogDir = Join-Path $Root 'logs'
$BackendUrl = "http://127.0.0.1:$BackendPort"
# Vite binds ::1 only, so localhost works and 127.0.0.1 does not.
$FrontendUrl = "http://localhost:$FrontendPort"

function Release-Port {
  param([int]$Port, [string]$Label)

  $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)

  if (-not $listeners) { return $false }

  foreach ($procId in @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
    if ($procId -eq $PID) {
      Write-Host "  $Label port $Port is held by this script; leaving it." -ForegroundColor DarkGray
      continue
    }

    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    $procName = if ($proc) { $proc.ProcessName } else { 'unknown' }

    Write-Host "  freeing $Label port $Port held by $procName (pid $procId)" -ForegroundColor Yellow
    & taskkill /PID $procId /T /F 2>&1 | Out-Null
  }

  # Give the kernel time to release the socket before anything rebinds it. A
  # uvicorn --reload tree is a parent plus a child process, and the socket is
  # only free once both have exited, which took over 5s in testing.
  for ($i = 0; $i -lt 40; $i++) {
    $still = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if (-not $still) { return $true }
    Start-Sleep -Milliseconds 250
  }

  Write-Host "  WARNING: port $Port is still bound after killing its owner." -ForegroundColor Red
  return $false
}

function Wait-HttpOk {
  param(
    [string]$Url,
    [int]$TimeoutSeconds = 120,
    [string]$Label = 'service'
  )

  # Readiness is an HTTP 200, never a TCP connect. Under --reload the reloader
  # parent binds the socket immediately while the child is still warming up, so
  # a connect succeeds tens of seconds before the app can answer a request.
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $elapsed = 0

  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
      if ($response.StatusCode -eq 200) { return $elapsed }
    }
    catch { }

    Start-Sleep -Seconds 1
    $elapsed++
  }

  Write-Host "  $Label did not answer $Url within ${TimeoutSeconds}s." -ForegroundColor Yellow
  return -1
}

if ($Logs) {
  New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

  $targets = @(
    (Join-Path $LogDir 'backend.out.log'),
    (Join-Path $LogDir 'backend.err.log'),
    (Join-Path $LogDir 'frontend.out.log'),
    (Join-Path $LogDir 'frontend.err.log')
  ) | Where-Object { Test-Path -LiteralPath $_ }

  if (-not $targets) {
    Write-Host "No logs yet in $LogDir. Start the services with .\dev.ps1 first." -ForegroundColor Yellow
    return
  }

  Write-Host "Following logs (Ctrl+C to stop following). Services keep running." -ForegroundColor Cyan
  Get-Content -Path $targets -Tail 20 -Wait
  return
}

Write-Host "Clearing stale listeners..." -ForegroundColor Cyan
Release-Port -Port $BackendPort -Label 'backend' | Out-Null
Release-Port -Port $FrontendPort -Label 'frontend' | Out-Null

if ($Stop) {
  Write-Host "Stopped. Ports $BackendPort and $FrontendPort are clear." -ForegroundColor Green
  return
}

$python = Join-Path $Root 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  Write-Error "Backend interpreter not found at $python. Create the venv first."
  return
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

Write-Host "Starting backend (uvicorn, --reload)..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $python `
  -ArgumentList '-m', 'uvicorn', 'backend.main:app', '--port', "$BackendPort", '--reload' `
  -WorkingDirectory $Root `
  -RedirectStandardOutput (Join-Path $LogDir 'backend.out.log') `
  -RedirectStandardError (Join-Path $LogDir 'backend.err.log') `
  -WindowStyle Hidden `
  -PassThru

# Startup imports the app and warms Ollama and Gemini, which takes 15-40s. The
# wait is bounded and reported rather than assumed, so a genuine failure is
# visible instead of looking like a hang.
Write-Host "  waiting for $BackendUrl/health (imports + model warm-up, 15-40s)..." -ForegroundColor DarkGray
$backendReady = Wait-HttpOk -Url "$BackendUrl/health" -TimeoutSeconds 150 -Label 'backend'

if ($backend.HasExited) {
  Write-Host "  backend exited during startup. Tail of backend.err.log:" -ForegroundColor Red
  Get-Content (Join-Path $LogDir 'backend.err.log') -Tail 20 | ForEach-Object { "    $_" }
  return
}

if ($backendReady -ge 0) {
  Write-Host "  backend answering on $BackendUrl after ~${backendReady}s" -ForegroundColor Green
}
else {
  Write-Host "  check: Get-Content logs\backend.err.log -Wait" -ForegroundColor DarkGray
}

Write-Host "Starting frontend (vite)..." -ForegroundColor Cyan
$npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) {
  Write-Error "npm not found on PATH."
  return
}

$frontend = Start-Process -FilePath $npm.Source `
  -ArgumentList 'run', 'dev' `
  -WorkingDirectory (Join-Path $Root 'frontend') `
  -RedirectStandardOutput (Join-Path $LogDir 'frontend.out.log') `
  -RedirectStandardError (Join-Path $LogDir 'frontend.err.log') `
  -WindowStyle Hidden `
  -PassThru

$frontendReady = Wait-HttpOk -Url "$FrontendUrl/" -TimeoutSeconds 60 -Label 'frontend'

if ($frontend.HasExited) {
  Write-Host "  frontend exited. Tail of frontend.err.log:" -ForegroundColor Red
  Get-Content (Join-Path $LogDir 'frontend.err.log') -Tail 20 | ForEach-Object { "    $_" }
}
elseif ($frontendReady -ge 0) {
  Write-Host "  frontend answering on $FrontendUrl after ~${frontendReady}s" -ForegroundColor Green
}

Write-Host ""
Write-Host "Open        : $FrontendUrl" -ForegroundColor White
Write-Host "  (use localhost, not 127.0.0.1 - vite binds the IPv6 loopback only)" -ForegroundColor DarkGray
Write-Host "Follow logs : .\dev.ps1 -Logs" -ForegroundColor DarkGray
Write-Host "Stop both   : .\dev.ps1 -Stop" -ForegroundColor DarkGray
Write-Host "PIDs        : backend $($backend.Id), frontend $($frontend.Id)" -ForegroundColor DarkGray