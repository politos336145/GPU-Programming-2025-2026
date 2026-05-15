#!/bin/bash

# build.sh - Linux/WSL native build script for Angry Santa (Vulkan + CUDA)
# Usage: ./build.sh

set -e

# ---- Paths ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
OUTPUT_DIR="$PROJECT_DIR/build"
SHADER_SRC="$PROJECT_DIR/src/renderers/shaders"

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo >&2 "cmake is required but it's not installed.  Aborting."; exit 1; }

# Se non ci sono i compilatori shader, disabilitiamo Vulkan
ENABLE_VULKAN="ON"
if command -v glslc >/dev/null 2>&1; then
    SHADER_COMPILER="glslc"
elif command -v glslangValidator >/dev/null 2>&1; then
    SHADER_COMPILER="glslangValidator"
else
    echo -e "\e[33mWarning: Neither glslc nor glslangValidator is installed. Building WITHOUT Vulkan renderer.\e[0m"
    ENABLE_VULKAN="OFF"
fi

# NVTX ranges for Nsight Systems profiling.
# Default ON to match the Windows build script.
# Can be overridden, e.g. ENABLE_NVTX=OFF ./linux/build.sh
ENABLE_NVTX="${ENABLE_NVTX:-ON}"

# CUDA line info for Nsight Compute source correlation.
# Can be overridden, e.g. ENABLE_CUDA_LINEINFO=OFF ./linux/build.sh
ENABLE_CUDA_LINEINFO="${ENABLE_CUDA_LINEINFO:-ON}"

# PTXAS verbose resource usage output (registers, spills, smem, cmem).
# Can be overridden, e.g. ENABLE_CUDA_PTXAS_VERBOSE=OFF ./linux/build.sh
ENABLE_CUDA_PTXAS_VERBOSE="${ENABLE_CUDA_PTXAS_VERBOSE:-ON}"

# ---- Clean & Configure ----
echo -e "\e[36m=== Configuring CMake ===\e[0m"
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# We use Ninja if available, otherwise default Makefiles
if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="-G Ninja"
else
    CMAKE_GENERATOR=""
fi

# Se nvcc è installato ma non è nel PATH standard, proviamo a trovarlo
if ! command -v nvcc >/dev/null 2>&1; then
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export CUDACXX="/usr/local/cuda/bin/nvcc"
    fi
fi

cmake .. \
    -DENABLE_VULKAN=$ENABLE_VULKAN \
    -DENABLE_NVTX=$ENABLE_NVTX \
    -DENABLE_CUDA_LINEINFO=$ENABLE_CUDA_LINEINFO \
    -DENABLE_CUDA_PTXAS_VERBOSE=$ENABLE_CUDA_PTXAS_VERBOSE \
    $CMAKE_GENERATOR \
    -DCMAKE_BUILD_TYPE=Release

# ---- Build ----
echo -e "\n\e[36m=== Building ===\e[0m"
cmake --build .

echo -e "\n\e[32m=== BUILD SUCCESSFUL ===\e[0m"
echo "Executable: $OUTPUT_DIR/AngrySanta_GPU"
echo "Arch: $ARCH"
echo "CMake options: ENABLE_VULKAN=$ENABLE_VULKAN ENABLE_NVTX=$ENABLE_NVTX ENABLE_CUDA_LINEINFO=$ENABLE_CUDA_LINEINFO ENABLE_CUDA_PTXAS_VERBOSE=$ENABLE_CUDA_PTXAS_VERBOSE"
echo "Run headless:    ./build/AngrySanta_GPU"
if [ "$ENABLE_VULKAN" = "ON" ]; then
    echo "Run with Vulkan: ./build/AngrySanta_GPU --render"
    echo ""
else
    echo -e "\n\e[33mNote: Built without Vulkan support. The --render flag will not work.\e[0m"
fi

cd "$PROJECT_DIR"
