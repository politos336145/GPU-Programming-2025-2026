# benchmark.ps1 - CPU vs GPU benchmark + physics scenarios + scaling
# Runs each scenario in both CPU-only and GPU mode, computes speedup.
# Usage: .\benchmark.ps1            (SISD + SIMD + GPU)
#        .\benchmark.ps1 --cpu      (SISD + SIMD only, no GPU)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$SISDexe = Join-Path $projectDir "SISD\build\AngrySanta_SISD.exe"
$SIMDexe = Join-Path $projectDir "SIMD\build\AngrySanta_SIMD.exe"
$GPUexe = Join-Path $projectDir "GPU\build\AngrySanta_GPU.exe"

$dateStr = Get-Date -Format "yyyyMMdd_HHmmss"
$outFile = Join-Path $scriptDir "benchmark_results_$dateStr.txt"

# Parse script-level flags
$runGPU = $true
if ($args -contains '--cpu') { $runGPU = $false }

if (-not (Test-Path $SISDexe)) {
  Write-Host "ERROR: $SISDexe not found. Run build.ps1 first." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path $SIMDexe)) {
  Write-Host "ERROR: $SIMDexe not found. Run build.ps1 first." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path $GPUexe)) {
  Write-Host "ERROR: $GPUexe not found. Run build.ps1 first." -ForegroundColor Red
  exit 1
}

function Write-Log {
  param([string]$Message, [ConsoleColor]$Color = [ConsoleColor]::White)
  Write-Host $Message -ForegroundColor $Color
  $Message | Out-File -FilePath $outFile -Append -Encoding UTF8
}

#  Run a single execution, capture all output, extract FPS
function Invoke-Run {
  param([string]$Label, [string]$ExtraArgs)

  Write-Log "    [$Label] AngrySanta_$Label.exe $ExtraArgs" -Color Gray

  $fps = 0.0
  $failed = $false
  
  if ($Label -eq "SISD") { $exe = $SISDexe }
  elseif ($Label -eq "SIMD") { $exe = $SIMDexe }
  elseif ($Label -eq "GPU") { $exe = $GPUexe }
  $cmd = "& `"$exe`" $ExtraArgs 2>&1"

  try {
    $output = Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { $failed = $true }
  } catch {
    $failed = $true
    $output = @()
  }

  foreach ($line in $output) {
    $s = "$line"   # coerce ErrorRecord to string
    if ($s -match '(^=|^  |^-|^Final|^CUDA Graph|^\[CUDA|^Mode|^GPU:|^Total program time)' -and $s -notmatch '^\s+(friction=|cell size=)') { Write-Log "        $s" }
    if ($s -match 'Effective FPS:\s+([\d.]+)') { $fps = [double]$Matches[1] }
    if ($s -match '(out of memory|CUDA error|cudaError|illegal memory|unspecified launch failure)') { $failed = $true }
  }
  if ($failed -or $fps -eq 0) { Write-Log "        ** RUN FAILED (process crashed or no FPS reported - possible OOM) **" -Color Red }
  
  Write-Log ""
  
  return $fps
}

#  Run a scenario in CPU + GPU mode, print comparison table
function Invoke-Comparison {
  param([string]$Label, [string]$ExtraArgs)

  Write-Log "$Label" -Color Cyan

  $cpuFps1 = Invoke-Run "SISD" "$ExtraArgs --no-log --no-trace --particles 100000"
  $cpuMs1 = if ($cpuFps1 -gt 0) { "{0:F3}" -f (1000.0 / $cpuFps1) } else { "N/A" }
  
  $cpuFps2 = Invoke-Run "SIMD" "$ExtraArgs --no-log --no-trace --particles 100000"
  $cpuMs2 = if ($cpuFps2 -gt 0) { "{0:F3}" -f (1000.0 / $cpuFps2) } else { "N/A" }
  
  $cpuSpeedup = if ($cpuFps2 -gt 0 -and $cpuFps1 -gt 0) { "{0:F1}x" -f ($cpuFps2 / $cpuFps1) } else { "---" }

  $gpuFps = 0.0
  if ($runGPU) {
    $gpuFps = Invoke-Run "GPU" "$ExtraArgs --no-log --no-trace --particles 100000"
    $gpuMs = if ($gpuFps -gt 0) { "{0:F3}" -f (1000.0 / $gpuFps) } else { "N/A" }
    $gpuSpeedup1 = if ($cpuFps1 -gt 0 -and $gpuFps -gt 0) { "{0:F1}x" -f ($gpuFps / $cpuFps1) } else { "---" }
    $gpuSpeedup2 = if ($cpuFps2 -gt 0 -and $gpuFps -gt 0) { "{0:F1}x" -f ($gpuFps / $cpuFps2) } else { "---" }
  }

  # Print comparison
  Write-Log "+---------------+-----------+-----------+-----------+"
  Write-Log "| Mode          |       FPS |  ms/frame |   Speedup |"
  Write-Log "+---------------+-----------+-----------+-----------+"
  Write-Log("| SISD          | {0,9:F1} | {1,9} |      1.0x |" -f $cpuFps1, $cpuMs1)
  Write-Log("| SIMD          | {0,9:F1} | {1,9} | {2,9} |" -f $cpuFps2, $cpuMs2, $cpuSpeedup)
  Write-Log "+---------------+-----------+-----------+-----------+"
  
  if ($runGPU) {
    Write-Log ""
    Write-Log "+---------------+-----------+-----------+-----------+"
    Write-Log "| Mode          |       FPS |  ms/frame |   Speedup |"
    Write-Log "+---------------+-----------+-----------+-----------+"
    Write-Log("| SISD          | {0,9:F1} | {1,9} |      1.0x |" -f $cpuFps1, $cpuMs1)
    Write-Log("| GPU           | {0,9:F1} | {1,9} | {2,9} |" -f $gpuFps, $gpuMs, $gpuSpeedup1)
    Write-Log "+---------------+-----------+-----------+-----------+"
    Write-Log ""
    Write-Log "+---------------+-----------+-----------+-----------+"
    Write-Log "| Mode          |       FPS |  ms/frame |   Speedup |"
    Write-Log "+---------------+-----------+-----------+-----------+"
    Write-Log("| SIMD          | {0,9:F1} | {1,9} |      1.0x |" -f $cpuFps2, $cpuMs2)
    Write-Log("| GPU           | {0,9:F1} | {1,9} | {2,9} |" -f $gpuFps, $gpuMs, $gpuSpeedup2)
    Write-Log "+---------------+-----------+-----------+-----------+"
  }  
}

Write-Log "==========================================================================================" -Color Yellow
Write-Log "=== ANGRY SANTA - CPU vs GPU BENCHMARK" -Color Yellow
Write-Log "=== Date: $dateStr" -Color Yellow
Write-Log "=== Mode: $(if($runGPU){'CPU + GPU'}else{'CPU only'})" -Color Yellow
Write-Log "==========================================================================================" -Color Yellow
Write-Log ""

# =====================================================================
#  SECTION 1 - Physics scenarios
# =====================================================================

Write-Log "=========================================================================================="
Write-Log "        SECTION 1 - PHYSICS SCENARIOS"
Write-Log "=========================================================================================="
Write-Log ""

Invoke-Comparison '1) Wet snow, high capture probability' ""
Write-Log ""
Invoke-Comparison '2) Dry snow - low adhesion, small ball' "--wetness-min 0.0 --wetness-max 0.3 --stick-k1 3.0"
Write-Log ""
Invoke-Comparison '3) Amplified avalanche feedback - high radius boost' "--stick-rboost 10.0"
Write-Log ""
Invoke-Comparison '4) High inter-particle friction - denser aggregation' "--part-friction 0.6"
Write-Log ""
Invoke-Comparison '5) Steep slope 45 deg - faster roll, harder captures' "--slope 45.0"
Write-Log ""

# Wait a bit between sections (useful for thermals / stability)
Start-Sleep -Seconds 5

# =====================================================================
#  SECTION 2 - Scaling benchmark (varying snowpack size)
# =====================================================================
$scalingConfigs = @(500000, 1000000, 5000000, 10000000)

Write-Log "=========================================================================================="
Write-Log "        SECTION 2 - SCALING BENCHMARK  (ball rolls to ground, varying snowpack sizes)"
Write-Log "                 Optimized CPU vs GPU across different snowpack sizes (N)"
Write-Log "=========================================================================================="
Write-Log ""
  
$scalingResults = @()
foreach ($N in $scalingConfigs) {
  Write-Log "--- Snowpack N=$N ---" -Color Cyan
  
  $cpuFps = Invoke-Run "SIMD" "--particles $N --no-trace --no-log"
  
  $gpuFps = 0.0
  if ($runGPU) { $gpuFps = Invoke-Run "GPU" "--particles $N --no-trace --no-log" }

  $speedup = if ($cpuFps -gt 0 -and $gpuFps -gt 0) { $gpuFps / $cpuFps } else { 0 }
  $scalingResults += [PSCustomObject]@{
    N       = $N
    CpuFps  = $cpuFps
    GpuFps  = $gpuFps
    Speedup = $speedup
  }
}

# Print scaling summary table
Write-Log "+-----------+-----------+-----------+-----------+"
Write-Log "| N         |  CPU FPS  |  GPU FPS  |   Speedup |"
Write-Log "+-----------+-----------+-----------+-----------+"
foreach ($r in $scalingResults) {
  $cf = if ($r.CpuFps -gt 0) { "{0:F1}" -f $r.CpuFps }  else { "---" }
  $gf = if ($r.GpuFps -gt 0) { "{0:F1}" -f $r.GpuFps }  else { "---" }
  $sp = if ($r.Speedup -gt 0) { "{0:F1}x" -f $r.Speedup } else { "---" }

  Write-Log ("| {0,9} | {1,9} | {2,9} | {3,9} |" -f $r.N, $cf, $gf, $sp)
}
Write-Log "+-----------+-----------+-----------+-----------+"
Write-Log ""