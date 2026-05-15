#!/bin/bash
# ============================================================================
# Compile GLSL shaders → SPIR-V for Vulkan
#
# Requires: glslc (from Vulkan SDK) or glslangValidator (from glslang-tools)
#
# Install on Jetson / Ubuntu:
#   sudo apt install glslang-tools    # provides glslangValidator
#   # or install Vulkan SDK for glslc
#
# Usage:
#   cd src/renderers/shaders
#   bash compile.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer glslc, fallback to glslangValidator
if command -v glslc &>/dev/null; then
    COMPILER="glslc"
    echo "Using glslc"
    glslc particle.vert -o particle_vert.spv
    glslc particle.frag -o particle_frag.spv
    glslc background.vert -o background_vert.spv
    glslc background.frag -o background_frag.spv
    glslc slope.vert -o slope_vert.spv
    glslc slope.frag -o slope_frag.spv
    echo "Compiled: particle_vert.spv, particle_frag.spv, background_vert.spv, background_frag.spv, slope_vert.spv, slope_frag.spv"

elif command -v glslangValidator &>/dev/null; then
    COMPILER="glslangValidator"
    echo "Using glslangValidator"
    glslangValidator -V particle.vert -o particle_vert.spv
    glslangValidator -V particle.frag -o particle_frag.spv
    glslangValidator -V background.vert -o background_vert.spv
    glslangValidator -V background.frag -o background_frag.spv
    glslangValidator -V slope.vert -o slope_vert.spv
    glslangValidator -V slope.frag -o slope_frag.spv
    echo "Compiled: particle_vert.spv, particle_frag.spv, background_vert.spv, background_frag.spv, slope_vert.spv, slope_frag.spv"

else
    echo "ERROR: No GLSL→SPIR-V compiler found."
    echo "Install one of:"
    echo "  sudo apt install glslang-tools"
    echo "  # or install Vulkan SDK (https://vulkan.lunarg.com)"
    exit 1
fi

echo ""
echo "SPIR-V files ready. Place them in the working directory or alongside the executable."
ls -la *.spv 2>/dev/null
