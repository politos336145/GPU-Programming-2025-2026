#!/bin/bash

# build.sh - Linux/WSL native build script for Angry Santa (SIMD version)
# Usage: ./build.sh

set -e

# ---- Paths ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo >&2 "cmake is required but it's not installed.  Aborting."; exit 1; }

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

cmake .. \
    $CMAKE_GENERATOR \
    -DCMAKE_BUILD_TYPE=Release

# ---- Build ----
echo -e "\n\e[36m=== Building ===\e[0m"
cmake --build .

echo -e "\n\e[32m=== BUILD SUCCESSFUL ===\e[0m"
echo "Executable: $BUILD_DIR/AngrySanta_SIMD"