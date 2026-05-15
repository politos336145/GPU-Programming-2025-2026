#!/bin/bash

# profile_nsys.sh - Linux/Jetson Nsight Systems capture helper
#
# Examples:
#   ./linux/profile_nsys.sh
#   ./linux/profile_nsys.sh -n 20000 -o nsys_N20000
#
# Notes:
# - Requires Nsight Systems CLI: `nsys` available in PATH.
# - Saves reports under: <project>/reports/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE="${PROJECT_DIR}/build/AngrySanta_GPU"
REPORTS_DIR="${PROJECT_DIR}/reports"

PARTICLES=100000
OUT_NAME="nsys_baseline_linux"
EXTRA_ARGS=("--no-log" "--no-trace")

usage() {
  cat <<EOF
Usage: $0 [options] [-- <extra AngrySanta_GPU args>]

Options:
  -n, --particles N   Particle count (default: ${PARTICLES})
  -o, --out NAME      Output basename (default: ${OUT_NAME})
  -h, --help          Show this help

Examples:
  $0
  $0 -n 20000 -o nsys_N20000
EOF
}

NO_EXTRA=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--particles) PARTICLES="$2"; shift 2;;
    -o|--out)       OUT_NAME="$2"; shift 2;;
    -h|--help)      usage; exit 0;;
    --)             shift; break;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${EXE}" ]]; then
  echo "ERROR: ${EXE} not found. Build first (e.g., ./linux/build.sh)." >&2
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: 'nsys' not found in PATH. Install Nsight Systems (CLI) or add it to PATH." >&2
  exit 1
fi

mkdir -p "${REPORTS_DIR}"
OUT_PATH="${REPORTS_DIR}/${OUT_NAME}"

SIM_ARGS=("--particles" "${PARTICLES}")
SIM_ARGS+=("${EXTRA_ARGS[@]}")

# Append passthrough args after --
if [[ $# -gt 0 ]]; then
  SIM_ARGS+=("$@")
fi

echo "=== Nsight Systems capture (Linux) ==="
echo "nsys : $(command -v nsys)"
echo "exe  : ${EXE}"
echo "out  : ${OUT_PATH}"
echo "args : ${SIM_ARGS[*]}"

nsys profile \
  -o "${OUT_PATH}" \
  --force-overwrite=true \
  --stats=true \
  --trace=cuda,nvtx \
  "${EXE}" "${SIM_ARGS[@]}"

echo "=== Done ==="
echo "Report saved under: ${REPORTS_DIR}"