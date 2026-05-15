#!/bin/bash

# benchmark.sh - CPU vs GPU benchmark + physics scenarios + scaling
# Runs each scenario in SISD, SIMD, and GPU mode, computes speedup.
# Usage: ./benchmark.sh            (SISD + SIMD + GPU)
#        ./benchmark.sh --cpu      (SISD + SIMD only, no GPU)

set -euo pipefail
export LC_NUMERIC=C   # force dot as decimal separator for printf/awk

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SISD_EXE="$PROJECT_DIR/SISD/build/AngrySanta_SISD"
SIMD_EXE="$PROJECT_DIR/SIMD/build/AngrySanta_SIMD"
GPU_EXE="$PROJECT_DIR/GPU/build/AngrySanta_GPU"

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$SCRIPT_DIR/benchmark_results_${DATE_STR}.txt"

# Parse script-level flags
RUN_GPU=1
for arg in "$@"; do
  if [[ "$arg" == "--cpu" ]]; then
    RUN_GPU=0
  fi
done

if [ ! -f "$SISD_EXE" ]; then
  echo -e "\e[31mERROR: $SISD_EXE not found. Run build.sh first.\e[0m"
  exit 1
fi

if [ ! -f "$SIMD_EXE" ]; then
  echo -e "\e[31mERROR: $SIMD_EXE not found. Run build.sh first.\e[0m"
  exit 1
fi

if [ ! -f "$GPU_EXE" ]; then
  echo -e "\e[31mERROR: $GPU_EXE not found. Run build.sh first.\e[0m"
  exit 1
fi

> "$OUT_FILE"

log()        { echo "$1" | tee -a "$OUT_FILE"; }
log_cyan()   { echo -e "\e[36m$1\e[0m"; echo "$1" >> "$OUT_FILE"; }
log_yellow() { echo -e "\e[33m$1\e[0m"; echo "$1" >> "$OUT_FILE"; }
log_red()    { echo -e "\e[31m$1\e[0m"; echo "$1" >> "$OUT_FILE"; }

ms_per_frame_from_fps() {
  local fps="$1"
  if awk "BEGIN{exit !($fps>0)}" >/dev/null 2>&1; then
    awk -v fps="$fps" 'BEGIN{printf "%.3f", 1000.0/fps}'
  else
    echo "N/A"
  fi
}

speedup_from_fps() {
  local faster="$1"
  local slower="$2"
  if awk "BEGIN{exit !($faster>0 && $slower>0)}" >/dev/null 2>&1; then
    awk -v f="$faster" -v s="$slower" 'BEGIN{printf "%.1fx", f/s}'
  else
    echo "---"
  fi
}

invoke_run() {
  local label="$1"
  shift
  local extra_args=("$@")

  log "    [$label] AngrySanta_$label ${extra_args[*]}" >&2

  local exe
  if   [[ "$label" == "SISD" ]]; then exe="$SISD_EXE"
  elif [[ "$label" == "SIMD" ]]; then exe="$SIMD_EXE"
  elif [[ "$label" == "GPU"  ]]; then exe="$GPU_EXE"
  fi

  local fps="0"
  local failed=0
  local tmpout
  tmpout=$(mktemp)

  local rc=0
  "$exe" "${extra_args[@]}" >"$tmpout" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then failed=1; fi

  while IFS= read -r line; do
    if [[ "$line" =~ ^= || "$line" =~ ^"  " || "$line" =~ ^- || "$line" =~ ^Final || "$line" =~ ^CUDA\ Graph || "$line" =~ ^\[CUDA || "$line" =~ ^Mode || "$line" =~ ^GPU: || "$line" =~ Effective\ FPS || "$line" =~ ^Total\ program\ time ]] && ! [[  "$line" =~ ^[[:space:]]+(friction=|cell\ size=) ]]; then
      log "        $line" >&2
    fi
    if [[ "$line" =~ Effective[[:space:]]FPS:[[:space:]]+([0-9]+\.?[0-9]*) ]]; then
      fps="${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ out\ of\ memory|CUDA\ error|cudaError|illegal\ memory|unspecified\ launch\ failure ]]; then
      failed=1
    fi
  done < "$tmpout"
  rm -f "$tmpout"

  if [[ "$failed" -eq 1 ]] || [[ "$fps" == "0" ]]; then
    log_red "        ** RUN FAILED (process crashed or no FPS reported - possible OOM) **" >&2
  fi

  log "" >&2
  echo "$fps"
}

invoke_comparison() {
  local label="$1"
  shift
  local extra_args=("$@")

  log_cyan "$label"

  local cpu_fps1
  cpu_fps1=$(invoke_run "SISD" "${extra_args[@]}" --no-log --no-trace --particles 100000)
  local cpu_ms1
  cpu_ms1=$(ms_per_frame_from_fps "$cpu_fps1")

  local cpu_fps2
  cpu_fps2=$(invoke_run "SIMD" "${extra_args[@]}" --no-log --no-trace --particles 100000)
  local cpu_ms2
  cpu_ms2=$(ms_per_frame_from_fps "$cpu_fps2")

  local cpu_speedup
  cpu_speedup=$(speedup_from_fps "$cpu_fps2" "$cpu_fps1")

  local gpu_fps="0"
  local gpu_ms="N/A"
  local gpu_speedup1="---"
  local gpu_speedup2="---"
  if [[ "$RUN_GPU" -eq 1 ]]; then
    gpu_fps=$(invoke_run "GPU" "${extra_args[@]}" --no-log --no-trace --particles 100000)
    gpu_ms=$(ms_per_frame_from_fps "$gpu_fps")
    gpu_speedup1=$(speedup_from_fps "$gpu_fps" "$cpu_fps1")
    gpu_speedup2=$(speedup_from_fps "$gpu_fps" "$cpu_fps2")
  fi

  log "+---------------+-----------+-----------+-----------+"
  log "| Mode          |       FPS |  ms/frame |   Speedup |"
  log "+---------------+-----------+-----------+-----------+"
  log "| SISD          | $(printf "%9.1f" "$cpu_fps1") | $(printf "%9s" "$cpu_ms1") |      1.0x |"
  log "| SIMD          | $(printf "%9.1f" "$cpu_fps2") | $(printf "%9s" "$cpu_ms2") | $(printf "%9s" "$cpu_speedup") |"
  log "+---------------+-----------+-----------+-----------+"

  if [[ "$RUN_GPU" -eq 1 ]]; then
    log ""
    log "+---------------+-----------+-----------+-----------+"
    log "| Mode          |       FPS |  ms/frame |   Speedup |"
    log "+---------------+-----------+-----------+-----------+"
    log "| SISD          | $(printf "%9.1f" "$cpu_fps1") | $(printf "%9s" "$cpu_ms1") |      1.0x |"
    log "| GPU           | $(printf "%9.1f" "$gpu_fps") | $(printf "%9s" "$gpu_ms") | $(printf "%9s" "$gpu_speedup1") |"
    log "+---------------+-----------+-----------+-----------+"
    log ""
    log "+---------------+-----------+-----------+-----------+"
    log "| Mode          |       FPS |  ms/frame |   Speedup |"
    log "+---------------+-----------+-----------+-----------+"
    log "| SIMD          | $(printf "%9.1f" "$cpu_fps2") | $(printf "%9s" "$cpu_ms2") |      1.0x |"
    log "| GPU           | $(printf "%9.1f" "$gpu_fps") | $(printf "%9s" "$gpu_ms") | $(printf "%9s" "$gpu_speedup2") |"
    log "+---------------+-----------+-----------+-----------+"
  fi

  log ""
}

MODE_STR="CPU + GPU"
if [[ "$RUN_GPU" -eq 0 ]]; then
  MODE_STR="CPU only"
fi

log_yellow "=========================================================================================="
log_yellow "=== ANGRY SANTA - CPU vs GPU BENCHMARK"
log_yellow "=== Date: $DATE_STR"
log_yellow "=== Mode: $MODE_STR"
log_yellow "=========================================================================================="
log ""

# =====================================================================
#  SECTION 1 - Physics scenarios
# =====================================================================

log "=========================================================================================="
log "        SECTION 1 - PHYSICS SCENARIOS"
log "=========================================================================================="
log ""

invoke_comparison "1) Wet snow, high capture probability"
log ""
invoke_comparison "2) Dry snow - low adhesion, small ball" --wetness-min 0.0 --wetness-max 0.3 --stick-k1 3.0
log ""
invoke_comparison "3) Amplified avalanche feedback - high radius boost" --stick-rboost 10.0
log ""
invoke_comparison "4) High inter-particle friction - denser aggregation" --part-friction 0.6
log ""
invoke_comparison "5) Steep slope 45 deg - faster roll, harder captures" --slope 45.0
log ""

# Wait a bit between sections (useful for thermals / stability)
sleep 5

# =====================================================================
#  SECTION 2 - Scaling benchmark (varying snowpack size)
# =====================================================================
SCALING_CONFIGS=(500000 1000000 5000000)

log "=========================================================================================="
log "        SECTION 2 - SCALING BENCHMARK  (ball rolls to ground, varying snowpack sizes)"
log "                 Optimized CPU vs GPU across different snowpack sizes (N)"
log "=========================================================================================="
log ""

SCALING_CPU_FPS=()
SCALING_GPU_FPS=()
SCALING_SPEEDUP=()

for N in "${SCALING_CONFIGS[@]}"; do
  log_cyan "--- Snowpack N=$N ---"

  cpu_fps=$(invoke_run "SIMD" --particles "$N" --no-trace --no-log)

  gpu_fps="0"
  if [[ "$RUN_GPU" -eq 1 ]]; then
    gpu_fps=$(invoke_run "GPU" --particles "$N" --no-trace --no-log)
  fi

  speedup="0"
  if [[ "$RUN_GPU" -eq 1 ]] && awk "BEGIN{exit !($cpu_fps>0 && $gpu_fps>0)}" >/dev/null 2>&1; then
    speedup=$(awk -v g="$gpu_fps" -v c="$cpu_fps" 'BEGIN{printf "%.1f", g/c}')
  fi

  SCALING_CPU_FPS+=("$cpu_fps")
  SCALING_GPU_FPS+=("$gpu_fps")
  SCALING_SPEEDUP+=("$speedup")
done

# Print scaling summary table
log "+-----------+-----------+-----------+-----------+"
log "| N         |  CPU FPS  |  GPU FPS  |   Speedup |"
log "+-----------+-----------+-----------+-----------+"

for i in "${!SCALING_CONFIGS[@]}"; do
  N="${SCALING_CONFIGS[$i]}"
  cf="${SCALING_CPU_FPS[$i]}"
  gf="${SCALING_GPU_FPS[$i]}"
  sp="${SCALING_SPEEDUP[$i]}"

  cf_out="---"
  gf_out="---"
  sp_out="---"

  if awk "BEGIN{exit !($cf>0)}" >/dev/null 2>&1; then
    cf_out=$(awk -v v="$cf" 'BEGIN{printf "%.1f", v}')
  fi
  if [[ "$RUN_GPU" -eq 1 ]] && awk "BEGIN{exit !($gf>0)}" >/dev/null 2>&1; then
    gf_out=$(awk -v v="$gf" 'BEGIN{printf "%.1f", v}')
  fi
  if [[ "$RUN_GPU" -eq 1 ]] && awk "BEGIN{exit !($sp>0)}" >/dev/null 2>&1; then
    sp_out="${sp}x"
  fi

  log "| $(printf "%9s" "$N") | $(printf "%9s" "$cf_out") | $(printf "%9s" "$gf_out") | $(printf "%9s" "$sp_out") |"
done

log "+-----------+-----------+-----------+-----------+"
log ""