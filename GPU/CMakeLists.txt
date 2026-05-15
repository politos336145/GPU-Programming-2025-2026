# ============================================================================
# Angry Santa - CMakeLists.txt (cross-platform: Windows + Linux/Jetson)
# ============================================================================
# Usage:
#   mkdir build && cd build
#
#   Headless (no renderer):
#     cmake .. && cmake --build . --config Release
#
#   With Vulkan renderer:
#     cmake .. -DENABLE_VULKAN=ON && cmake --build . --config Release
#
#   Run:
#     ./AngrySanta_GPU                      (headless)
#     ./AngrySanta_GPU --render             (with Vulkan)
# ============================================================================

cmake_minimum_required(VERSION 3.10)
project(AngrySanta_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# --- Options -----------------------------------------------------------------
option(ENABLE_VULKAN "Build with Vulkan renderer (requires Vulkan SDK + GLFW)" OFF)
option(ENABLE_NVTX   "Enable NVTX ranges for Nsight Systems profiling"        OFF)
option(ENABLE_CUDA_LINEINFO "Enable CUDA -lineinfo for Nsight Compute/source correlation" OFF)
option(ENABLE_CUDA_PTXAS_VERBOSE "Enable CUDA ptxas verbose resource usage output (-Xptxas=-v)" OFF)

# Detect target architecture
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CPU_ARCH_LOWER)
set(CPU_ARCH_X86 OFF)
set(CPU_ARCH_ARM OFF)
if(CPU_ARCH_LOWER MATCHES "^(x86_64|amd64|x86|i[3-6]86)$")
    set(CPU_ARCH_X86 ON)
elseif(CPU_ARCH_LOWER MATCHES "^(aarch64|arm64|armv[0-9]+|arm)$")
    set(CPU_ARCH_ARM ON)
endif()
message(STATUS "Host CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# --- CUDA setup --------------------------------------------------------------
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
    find_package(CUDAToolkit REQUIRED)
endif()

# Default GPU architecture - auto-detect platform.
# - Desktop (x86_64): sm_50 (GTX 960M baseline)
# - Jetson  (aarch64): sm_53
# Override at configure time: -DCMAKE_CUDA_ARCHITECTURES=75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
        set(CMAKE_CUDA_ARCHITECTURES 53 CACHE STRING "CUDA architectures")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 50 CACHE STRING "CUDA architectures")
    endif()
endif()

# CMAKE_CUDA_ARCHITECTURES is only honoured by CMake >= 3.18.
# On older toolchains (e.g. JetPack 4.x ships CMake 3.10) we must pass the
# -gencode flag manually, otherwise nvcc falls back to the default sm_30.
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CMake < 3.18: manually added -gencode for sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Separable compilation OFF - d_params (__constant__) must be shared across all
# .cu files in a single translation unit.  gpu_unity.cu #includes the individual
# .cu files so they compile as one unit (main.cu is compiled separately).
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# --- Source files ------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHARED_SRC_DIR ${CMAKE_SOURCE_DIR}/../shared)
set(SHARED_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../shared/include)

set(SOURCES
    ${SHARED_SRC_DIR}/cli.cpp
    ${SHARED_SRC_DIR}/global.cpp
    ${SHARED_SRC_DIR}/logging.cpp
    ${SRC_DIR}/main.cu
)

# __constant__ d_params must live in a single translation unit.
# gpu_unity.cu manually #includes grid.cu, simulation.cu, snowpack.cu so that
# all kernels share the same d_params instance.  main.cu stays separate (it
# has no device kernels that read d_params).
list(APPEND SOURCES ${SRC_DIR}/gpu_unity.cu)

set(HEADERS
    ${SHARED_INCLUDE_DIR}/cli.h
    ${SHARED_INCLUDE_DIR}/global.h
    ${SHARED_INCLUDE_DIR}/logging.h
    ${SHARED_INCLUDE_DIR}/profiling.h
    ${SHARED_INCLUDE_DIR}/types.h
    ${SRC_DIR}/include/grid.cuh
    ${SRC_DIR}/include/helpers.cuh
    ${SRC_DIR}/include/kernel.cuh
    ${SRC_DIR}/include/simulation.cuh
    ${SRC_DIR}/include/snowpack.cuh
)

# --- Vulkan renderer sources (optional) --------------------------------------
if(ENABLE_VULKAN)
    list(APPEND SOURCES ${SRC_DIR}/renderers/vulkan_renderer.cu)
    list(APPEND SOURCES ${SRC_DIR}/renderers/stb_image_impl.cpp)
    list(APPEND SOURCES ${SRC_DIR}/renderers/imgui_impl_all.cpp)
    list(APPEND HEADERS ${SRC_DIR}/renderers/vulkan_renderer.h)
    list(APPEND HEADERS ${SRC_DIR}/renderers/stb_image.h)
endif()

# --- Executable --------------------------------------------------------------
add_executable(AngrySanta_GPU ${SOURCES} ${HEADERS})

target_include_directories(AngrySanta_GPU PRIVATE ${SRC_DIR})

# CUB is included in CUDA Toolkit starting from 11.0.
# For CUDA 10.2, we need to download it or use a local copy.
# We'll use ExternalProject to download CUB 1.8.0 (compatible with CUDA 10.2)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    include(ExternalProject)
    set(CUB_PREFIX "${CMAKE_BINARY_DIR}/cub-download")
    ExternalProject_Add(cub_download
        PREFIX ${CUB_PREFIX}
        URL https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    # The source directory where CUB will be extracted
    set(CUB_SOURCE_DIR "${CUB_PREFIX}/src/cub_download")
    target_include_directories(AngrySanta_GPU PRIVATE ${CUB_SOURCE_DIR})
    add_dependencies(AngrySanta_GPU cub_download)
endif()

# CUDA flags
target_compile_options(AngrySanta_GPU PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

if(ENABLE_CUDA_LINEINFO)
    target_compile_options(AngrySanta_GPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
    message(STATUS "CUDA line info enabled (-lineinfo)")
endif()

if(ENABLE_CUDA_PTXAS_VERBOSE)
    target_compile_options(AngrySanta_GPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    message(STATUS "CUDA ptxas verbose output enabled (-Xptxas=-v)")
endif()

# --- MSVC / CUDA STL compatibility ------------------------------------------
# MSVC 17.10+ STL requires CUDA >= 12.4.  When building with an older toolkit
# we suppress the static_assert so the compiler pair still works.
if(MSVC AND CUDAToolkit_VERSION VERSION_LESS "12.4")
    target_compile_definitions(AngrySanta_GPU PRIVATE
        _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    )
    # The same define must reach the host compiler invoked by nvcc
    target_compile_options(AngrySanta_GPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
    )
endif()

# --- NVTX profiling support --------------------------------------------------
if(ENABLE_NVTX)
    target_compile_definitions(AngrySanta_GPU PRIVATE ENABLE_NVTX)
    if(TARGET CUDA::nvToolsExt)
        target_link_libraries(AngrySanta_GPU PRIVATE CUDA::nvToolsExt)
    else()
        # Fallback: find nvToolsExt manually
        find_library(NVTX_LIB nvToolsExt
            HINTS ${CUDAToolkit_LIBRARY_DIR}
                  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                  ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
        if(NVTX_LIB)
            target_link_libraries(AngrySanta_GPU PRIVATE ${NVTX_LIB})
        else()
            message(WARNING "NVTX library not found - NVTX ranges will be no-ops")
        endif()
    endif()
endif()

# --- MSVC portability defines ------------------------------------------------
if(MSVC)
    target_compile_definitions(AngrySanta_GPU PRIVATE
        NOMINMAX             # prevent min/max macros from windows.h
    )
endif()

# --- Release / Debug ---------------------------------------------------------
target_compile_definitions(AngrySanta_GPU PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:DEBUG>
)

# --- Vulkan renderer dependencies -------------------------------------------
if(ENABLE_VULKAN)
    target_compile_definitions(AngrySanta_GPU PRIVATE ENABLE_VULKAN)

    # Vulkan
    find_package(Vulkan REQUIRED)
    target_link_libraries(AngrySanta_GPU PRIVATE Vulkan::Vulkan)

    # ImGui
    target_include_directories(AngrySanta_GPU PRIVATE ${SRC_DIR}/renderers/imgui)
    target_include_directories(AngrySanta_GPU PRIVATE ${SRC_DIR}/renderers/imgui/backends)

    # GLFW
    if(WIN32)
        # Windows: use pre-compiled GLFW from C:\glfw-3.4
        set(GLFW_ROOT "C:/glfw-3.4" CACHE PATH "GLFW root directory")
        target_include_directories(AngrySanta_GPU PRIVATE ${GLFW_ROOT}/include)

        # Pick the right lib folder for the MSVC version
        if(EXISTS "${GLFW_ROOT}/lib-vc2022")
            target_link_directories(AngrySanta_GPU PRIVATE ${GLFW_ROOT}/lib-vc2022)
        elseif(EXISTS "${GLFW_ROOT}/lib-vc2019")
            target_link_directories(AngrySanta_GPU PRIVATE ${GLFW_ROOT}/lib-vc2019)
        endif()
        target_link_libraries(AngrySanta_GPU PRIVATE glfw3)
    else()
        # Linux / Jetson: use system GLFW (apt install libglfw3-dev)
        find_package(glfw3 REQUIRED)
        target_link_libraries(AngrySanta_GPU PRIVATE glfw)
    endif()

    # Shader compilation
    find_program(GLSLC glslc REQUIRED)
    file(GLOB SHADER_FILES
        ${SRC_DIR}/renderers/shaders/*.vert
        ${SRC_DIR}/renderers/shaders/*.frag
        ${SRC_DIR}/renderers/shaders/*.comp
    )

    foreach(SHADER ${SHADER_FILES})
        get_filename_component(FILE_NAME ${SHADER} NAME)
        string(REPLACE "." "_" SPV_NAME ${FILE_NAME})

        set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
            COMMAND glslc ${SHADER} -o ${SPV}
            DEPENDS ${SHADER}
        )
        list(APPEND SPV_SHADERS ${SPV})
    endforeach()

    add_custom_target(compile_shaders ALL DEPENDS ${SPV_SHADERS})
    add_dependencies(AngrySanta_GPU compile_shaders)

    add_custom_command(TARGET AngrySanta_GPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${SRC_DIR}/renderers/textures
            ${CMAKE_BINARY_DIR}/textures
    )
endif()

# --- Link against math library (required on Linux) ----------------------------
if(NOT MSVC)
    target_link_libraries(AngrySanta_GPU PRIVATE m)
endif()

target_compile_definitions(AngrySanta_GPU PRIVATE EXEPATH="${CMAKE_BINARY_DIR}/")