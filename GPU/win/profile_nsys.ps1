# profile_nsys.ps1 - Windows Nsight Systems capture helper
# Usage examples:
#   .\profile_nsys.ps1
#   .\profile_nsys.ps1 -Particles 20000 -Seed 42 -OutName nsys_N20000

param(
  [int]$Particles = 1000000,
  [string]$OutName = "nsys_baseline_win",
  [string]$ExtraArgs = "--no-log --no-trace"
)

$ErrorActionPreference = "Stop"

$winDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $winDir
$exe = Join-Path $projectDir "build\AngrySanta_GPU.exe"
$reportsDir = "reports"

if (-not (Test-Path $exe)) {
  Write-Host "ERROR: $exe not found. Run win\\build.ps1 first." -ForegroundColor Red
  exit 1
}

New-Item -ItemType Directory -Force $reportsDir | Out-Null

# Nsight Systems is installed on this machine but typically not in PATH.
# Default expected location (as detected on this setup):
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\target-windows-x64\nsys.exe"

if (-not (Test-Path $nsys)) {
  # Fallback: try to locate it under Program Files
  $candidate = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Systems*" -Recurse -Filter nsys.exe -ErrorAction SilentlyContinue |
               Select-Object -First 1 -ExpandProperty FullName
  if ($candidate) {
    $nsys = $candidate
  } else {
    Write-Host "ERROR: nsys.exe not found. Install Nsight Systems or add it to PATH." -ForegroundColor Red
    exit 1
  }
}

$outPath = Join-Path $reportsDir $OutName

Write-Host "=== Nsight Systems capture (Windows) ===" -ForegroundColor Cyan
Write-Host "nsys   : $nsys"
Write-Host "exe    : $exe"
Write-Host "out    : $outPath"
Write-Host "args   : --particles $Particles $ExtraArgs"

# Build argument array for safe invocation
$simArgs = @(
  "--particles", "$Particles"
)
if ($ExtraArgs -and $ExtraArgs.Trim().Length -gt 0) {
  $simArgs += ($ExtraArgs -split "\s+")
}

& $nsys profile `
  -o $outPath `
  --force-overwrite=true `
  --stats=true `
  --trace=cuda,nvtx `
  $exe @simArgs

Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Report saved under: $reportsDir"