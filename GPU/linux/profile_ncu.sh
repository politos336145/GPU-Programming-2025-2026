#!/bin/bash

# profile_ncu.sh - Nsight Compute per-kernel profiling (memory throughput + branch divergence)
# Run on the machine with the GPU (e.g. Jetson Nano, GTX 960M).
#
# Usage:
#   ./linux/profile_ncu.sh                                    # all kernels
#   ./linux/profile_ncu.sh -k neighborCollision               # single kernel
#   ./linux/profile_ncu.sh -o ncu_after_opt                   # custom output name
#   ./linux/profile_ncu.sh -k forceTether -l 10               # 10 launches

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE="${PROJECT_DIR}/build/AngrySanta_GPU"
REPORTS_DIR="${PROJECT_DIR}/reports"

PARTICLES=100000
OUT_NAME="ncu_profile"
KERNEL=""
LAUNCH_COUNT=5
EXTRA_ARGS=("--no-log" "--no-trace")

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -n, --particles N    Particle count (default: ${PARTICLES})
  -o, --out NAME       Output basename (default: ${OUT_NAME})
  -k, --kernel REGEX   Filter to kernel name substring (default: all)
  -l, --launches N     Number of kernel launches to capture (default: ${LAUNCH_COUNT})
  -h, --help           Show this help

Metrics collected:
  Memory throughput:  DRAM/L1/L2 bandwidth %, load/store efficiency, bytes R/W
  Branch divergence:  uniform branch %, predicated-on/off instruction counts
  Occupancy:          active warps %, warps per cycle
  Compute:            FP32 ADD/MUL/FMA instruction counts

Examples:
  $0
  $0 -k neighborCollision -o ncu_collision
  $0 -k forceTether -l 10
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--particles) PARTICLES="$2"; shift 2;;
    -o|--out)       OUT_NAME="$2"; shift 2;;
    -k|--kernel)    KERNEL="$2"; shift 2;;
    -l|--launches)  LAUNCH_COUNT="$2"; shift 2;;
    -h|--help)      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# --- Validate exe ---
if [[ ! -f "${EXE}" ]]; then
  echo "ERROR: ${EXE} not found. Build first (e.g., ./linux/build.sh)." >&2
  exit 1
fi

# --- Locate ncu ---
NCU=""
if command -v ncu >/dev/null 2>&1; then
  NCU="$(command -v ncu)"
elif [[ -n "${CUDA_PATH:-}" ]] && [[ -x "${CUDA_PATH}/bin/ncu" ]]; then
  NCU="${CUDA_PATH}/bin/ncu"
elif [[ -x "/usr/local/cuda/bin/ncu" ]]; then
  NCU="/usr/local/cuda/bin/ncu"
else
  echo "ERROR: 'ncu' not found. Install Nsight Compute or add it to PATH." >&2
  echo "       On Jetson: sudo apt install nsight-compute" >&2
  exit 1
fi

mkdir -p "${REPORTS_DIR}"
OUT_PATH="${REPORTS_DIR}/${OUT_NAME}"

# --- Metrics ---
METRICS=$(cat <<'METRICS_END'
dram__throughput.avg.pct_of_peak_sustained_elapsed,
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,
lts__throughput.avg.pct_of_peak_sustained_elapsed,
sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,
sm__sass_average_data_bytes_per_sector_mem_global_op_st.pct,
dram__bytes_read.sum,
dram__bytes_write.sum,
smsp__sass_average_branch_targets_threads_uniform.pct,
smsp__thread_inst_executed_pred_on.sum,
smsp__thread_inst_executed_pred_off.sum,
sm__warps_active.avg.pct_of_peak_sustained_elapsed,
sm__warps_active.avg.per_cycle_active,
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,
sm__sass_thread_inst_executed_op_ffma_pred_on.sum
METRICS_END
)
# Remove newlines
METRICS=$(echo "${METRICS}" | tr -d '\n' | tr -d ' ')

# --- Simulation args ---
SIM_ARGS=("--particles" "${PARTICLES}" "${EXTRA_ARGS[@]}")

echo "=== Nsight Compute kernel profiling ==="
echo "ncu    : ${NCU}"
echo "exe    : ${EXE}"
echo "out    : ${OUT_PATH}"
echo ""

# --- Build ncu args ---
NCU_ARGS=(
  "--set" "full"
  "--metrics" "${METRICS}"
  "--launch-skip" "0"
  "--launch-count" "${LAUNCH_COUNT}"
  "-f"
  "-o" "${OUT_PATH}"
)

if [[ -n "${KERNEL}" ]]; then
  NCU_ARGS+=("--kernel-name" "regex:${KERNEL}")
  echo "Kernel filter: ${KERNEL}"
else
  echo "Profiling ALL kernels (first ${LAUNCH_COUNT} launches each)"
fi
echo ""

# --- On Jetson, ncu may need sudo ---
if [[ -f /etc/nv_tegra_release ]] 2>/dev/null; then
  echo "(Jetson detected - running ncu with sudo)"
  sudo "${NCU}" "${NCU_ARGS[@]}" "${EXE}" "${SIM_ARGS[@]}"
else
  "${NCU}" "${NCU_ARGS[@]}" "${EXE}" "${SIM_ARGS[@]}"
fi

echo ""
echo "=== Done ==="
echo "Report: ${OUT_PATH}.ncu-rep"
echo ""
echo "Quick summary:"

# --- Print text summary ---
"${NCU}" --import "${OUT_PATH}.ncu-rep" --page raw --csv 2>/dev/null | \
  head -20 || echo "(Install ncu-ui to view full report: ncu-ui ${OUT_PATH}.ncu-rep)"

echo ""
echo "Open full report:  ncu-ui ${OUT_PATH}.ncu-rep"
