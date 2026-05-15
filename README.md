# Angry Santa: a Snowball Simulation

**Institution**: Politecnico di Torino (Academic Year 2025/2026)<br>
**Student**: Maurizio Grisotto (id 336145)

This project demonstrates advanced CUDA programming techniques for DEM-based particle physics simulation, emphasizing GPU architecture utilization, memory hierarchy optimization, and performance profiling.

The GPU implementation is compared against CPUs-only version (SISD and SIMD + multi-threading optimizations).

---

## 1. Story

We are approaching the Christmas season and Santa Claus is getting tired and angry because his elves team is behind schedule. In a moment of pure frustration, he decides to destroy the entire village with a gigantic rolling snowball.

So… let's simulate this physical phenomenon using **GPU parallel computing**!

![Angry Santa](docs/01%20-%20Pitch/Angry%20Santa.png)

---

## 2. Quick Start

Each version (GPU and CPU) has its own executable but supports the same command-line interface (CLI) for configuring simulation parameters, with sensible defaults for all options. Run `AngrySanta_XXXX.exe --help` for a full list of options.

You need to build each version (see instructions below) to get the executables in the `build` folder.

If you want to execute the .sh scripts on Linux, make sure to set the executable flag and convert line endings:

1) sed -i 's/\r$//' nomefile.sh
2) chmod +x nomefile.sh
3) ./nomefile.sh

### 2.1. Build and Run (Windows - VS2022 + CUDA)
```powershell
.\win\build.ps1
```

You need to have Visual Studio 2022 with the Desktop Development with C++ workload, the CUDA Toolkit (10.2 or higher), Windows 10 Kit and Vulkan SDK installed. The build script will compile the project in Release mode.
Fix the paths in the script if your installation directories differ.

```powershell
# ---- MODIFY HERE: Paths to tools & SDK ----
$cmakePath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninjaPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$msvcRoot = "C:\Program Files\Microsoft Visual Studio\2022\Professional"
$msvc = "$msvcRoot\VC\Tools\MSVC\14.44.35207"
$msvcBin = "$msvc\bin\HostX64\x64"
$msvcInclude = "$msvc\include"
$msvcLib = "$msvc\lib\x64"
$windowsSdkRoot = "C:\Program Files (x86)\Windows Kits\10"
$windowsSdkBin = "$windowsSdkRoot\bin\10.0.26100.0\x64"
$windowsSdkLib = "$windowsSdkRoot\Lib\10.0.26100.0\ucrt\x64;$windowsSdkRoot\Lib\10.0.26100.0\um\x64"
$windowsSdkInclude = "$windowsSdkRoot\Include\10.0.26100.0\ucrt;$windowsSdkRoot\Include\10.0.26100.0\um;$windowsSdkRoot\Include\10.0.26100.0\shared"
$cudaPath  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$vulkanSdk = "C:\VulkanSDK\1.4.341.1"
```

### 2.2. Build and Run (Linux - GCC + CUDA)
```bash
./linux/build.sh
```

Make sure you have GCC 7+ and CUDA Toolkit 10.2 or higher installed. The build script will compile the project in Release mode using CMake and Ninja.
If you want to enable the Vulkan renderer, install the Vulkan SDK and make sure `glslc` or `glslangValidator` is in your PATH.

---

## 3. Benchmarking

In the `benchmark` folder you can find scripts for comparing GPU vs CPU performance and for running scaling benchmarks with different particle counts. The scripts will run the executables with predefined CLI options and log the results in TXT files for later analysis.

### 3.1. Physics Scenarios
1) Wet snow, high capture probability ("")
2) Dry snow - low adhesion, small ball ("--wetness-min 0.0 --wetness-max 0.3 --stick-k1 3.0")
3) Amplified avalanche feedback - high radius boost ("--stick-rboost 10.0")
4) High inter-particle friction - denser aggregation ("--part-friction 0.6")
5) Steep slope 45 deg - faster roll, harder captures ("--slope 45.0")

### 3.2. Scaling Benchmarks
Run the scaling benchmarks to analyze how the execution time changes with increasing particle counts. The scripts will execute the GPU and CPU versions with a range of particle counts (500k, 1M, 5M, 10M not on the Jetson Nano) and log the results for later plotting.

---

## 4. Key Features

- **Three-subsystem architecture**: Snowpack (static terrain) + Shell (dynamic DEM particles) + Core (rigid body)
- **Soft-sphere collision model** with contact forces, Coulomb friction, and cohesion
- **Probabilistic particle capture** via sigmoid sticking model with avalanche feedback
- **Spatial hashing** with CUB radix sort for O(N) collision detection
- **Advanced CUDA optimizations**:
  - Structure-of-Arrays (SoA) layout for coalesced memory access
  - `__constant__` memory for simulation parameters
  - Shared-memory tiled neighbor traversal
  - `__ldg()` read-only data cache routing
  - Multi-stream physics pipeline (simStream + gridStream with fork/join)

  - Sparse grid clearing (clears only the cells used last frame; avoids full `cellStart/cellEnd` memset)
  - Pinned memory + async stream for trace copies
  - Atomic lock-free slot reservation for particle capture
  - Unity Build for global `__constant__` symbol visibility
- **Optional Vulkan real-time renderer** with CUDA<->Vulkan zero-copy interop

---

## 5. Project Structure

```
Angry Santa/
├── Project Assignement 2025-26.pdf
├── .gitignore
├── .vscode/
├── README.md                                         # This file
│
├── docs/
│     ├── TECHNICAL_README.md                         # Detailed technical docs
│     ├── 01 - Pitch/                                 # Delivery 01: Project pitch
│     ├── 02 - Proposal/                              # Delivery 02: Proposal document
│     ├── 03 - Report/                                # Delivery 03: Final report
│     └── 04 - Presentation/                          # Delivery 04: Presentation slides
│
├── GPU/                                              # NVIDIA CUDA implementation (accelerated version)
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── gpu_unity.cu                          # Unity build compilation unit
│     │     ├── main.cu                               # Entry point, simulation loop, CLI parsing
│     │     ├── simulation.cu                         # GPU host functions + wrappers
│     │     ├── grid.cu                               # Spatial grid (hash, CUB sort, cell ranges)
│     │     ├── snowpack.cu                           # Snowpack init + probabilistic capture kernel
│     │     ├── include/
│     │     │     ├── simulation.cuh                  # GPU simulation declarations
│     │     │     ├── grid.cuh                        # Grid host function declarations
│     │     │     ├── snowpack.cuh                    # Snowpack host function declarations
│     │     │     ├── kernel.cuh                      # All __global__ kernel declarations
│     │     │     ├── helpers.cuh                     # CUDA_CHECK, constant memory, helpers
│     │     └── renderers/
│     │           ├── vulkan_renderer.cu              # Vulkan + CUDA interop renderer
│     │           ├── vulkan_renderer.h               # Renderer API
│     │           ├── vulkan_renderer_billboard.inl   # Billboard rendering implementation
│     │           ├── imgui_iml_all.cpp               # ImGui implementation (GLFW + Vulkan)
│     │           ├── stb_image_impl.cpp              # stb_image implementation for texture loading
│     │           ├── stb_image.h                     # stb_image header
|     |           ├── imgui/                          # Dear ImGui library for UI (submodule or copy)
│     │           ├── shaders/                        # GLSL + SPIR-V shaders
│     │           └── textures/                       # Texture images
│     ├── win/
│     │     ├── build.ps1                             
│     │     └── profile_nsys.ps1                      # Nsight Systems profiling
│     └── linux/
│           ├── build.sh                              
│           ├── profile_ncu.sh                        # Nsight Compute profiling
│           └── profile_nsys.sh                       # Nsight Systems profiling                                
│
├── SIMD/                                             # OpenMP multi-threaded SIMD version (SSE-vectorized)
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── main.cpp
│     │     └── simulation.cpp
│     ├── win/
│     │     └── build.ps1
│     └── linux/
│           └── build.sh
│
├── SISD/                                             # Single-threaded CPU reference implementation
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── main.cpp
│     │     └── simulation.cpp
│     ├── win/
│     │     └── build.ps1
│     └── linux/
│           └── build.sh
│
├── benchmark/
│     ├── benchmark.ps1                               # Windows scaling benchmark script
│     └── benchmark.sh                                # Linux scaling benchmark script
│
└── shared/                                           # Shared code across all implementations
      ├── cli.cpp                                     # CLI argument parsing
      ├── global.cpp                                  # Global functions used by all versions
      ├── logging.cpp                                 # CSV + stdout logging
      ├── simulation_CPU.cpp                          # Common CPU simulation routines
      └── include/
            ├── cli.h                                 # CLI parsing declarations
            ├── global.h                              # Global function declarations
            ├── logging.h                             # Logging function declarations
            ├── memory_CPU.h                          # CPU memory management declarations
            ├── profiling.h                           # Profiling utilities declarations
            ├── simulation_CPU.h                      # CPU simulation declarations
            └── types.h                               # Common types and constants
```

---

## 6. Requirements

### 6.1. Hardware Tested
**NVIDIA GPU** with Compute Capability **5.0+** (Maxwell through Ada Lovelace) and minimum 2 GB GPU memory recommended

- Jetson Nano: CPU (SoC) ARM Cortex-A57 (4 cores, 4 threads), GPU Tegra X1 (128 CUDA cores, SM 5.3 – VRAM 4GB), Linux
- Laptop: CPU Intel i7-6700HQ (4 cores, 8 threads), GPU GeForce GTX 960M (640 CUDA cores, SM 5.0 – VRAM 4GB), Windows

### 6.2. Software
- **CUDA Toolkit** 10.2 or higher (11.x+ recommended for built-in CUB)
- **CMake** 3.10+ (3.16+ recommended for Unity Build)
- **C++14** compiler (g++ 7+, MSVC 2017+)
- **Ninja** build backend (recommended); **Make** is supported on Linux

### 6.3. Optional
- **Vulkan SDK** + **GLFW 3.x** for real-time rendering
- **Nsight Systems** / **Nsight Compute** for profiling

---

## 7. References

- NVIDIA CUDA Toolkit docs: https://docs.nvidia.com/cuda/
- NVIDIA Nsight Systems & Compute: https://developer.nvidia.com/tools-overview
- CUB Library: https://nvidia.github.io/cccl/unstable/cub/index.html
- Vulkan API Specification: https://vulkan.lunarg.com/sdk/home
- Dear Imgui API Reference: https://github.com/ocornut/imgui
- Cundall & Strack (1979): "A discrete numerical model for granular assemblies"
- Teschner et al. (2003): "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
- Poschel & Schwager (2005): "Computational Granular Dynamics"
- Green (2010): "Particle Simulation using CUDA"