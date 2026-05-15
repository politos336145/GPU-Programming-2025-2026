cmake_minimum_required(VERSION 3.10)
project(AngrySanta_SIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# --- Source files ------------------------------------------------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHARED_SRC_DIR ${CMAKE_SOURCE_DIR}/../shared)
set(SHARED_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../shared/include)

set(SOURCES
    ${SHARED_SRC_DIR}/cli.cpp
    ${SHARED_SRC_DIR}/global.cpp
    ${SHARED_SRC_DIR}/logging.cpp
    ${SRC_DIR}/main.cpp
    ${SHARED_SRC_DIR}/simulation_CPU.cpp
    ${SRC_DIR}/simulation.cpp
)

set(HEADERS
    ${SHARED_INCLUDE_DIR}/cli.h
    ${SHARED_INCLUDE_DIR}/global.h
    ${SHARED_INCLUDE_DIR}/logging.h
    ${SHARED_INCLUDE_DIR}/memory_CPU.h
    ${SHARED_INCLUDE_DIR}/profiling.h
    ${SHARED_INCLUDE_DIR}/simulation_CPU.h
    ${SHARED_INCLUDE_DIR}/types.h
)

# --- Executable --------------------------------------------------------------
add_executable(AngrySanta_SIMD ${SOURCES} ${HEADERS})

target_include_directories(AngrySanta_SIMD PRIVATE ${SRC_DIR} ${SHARED_SRC_DIR} ${SHARED_INCLUDE_DIR})
target_compile_definitions(AngrySanta_SIMD PRIVATE SIMD)

# --- OpenMP ------------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(AngrySanta_SIMD PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(AngrySanta_SIMD PRIVATE HAS_OPENMP)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - running single-threaded")
endif()

# --- Link against math library (required on Linux) ----------------------------
if(NOT MSVC)
    target_link_libraries(AngrySanta_SIMD PRIVATE m)
endif()

# --- Architecture detection --------------------------------------------------
include(CheckCXXCompilerFlag)

# Detect target architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(TARGET_ARCH "ARM64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
    set(TARGET_ARCH "ARM32")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64")
    set(TARGET_ARCH "X86_64")
else()
    set(TARGET_ARCH "GENERIC")
endif()

message(STATUS "Detected architecture: ${TARGET_ARCH} (${CMAKE_SYSTEM_PROCESSOR})")

# --- MSVC portability defines ------------------------------------------------
if(MSVC)
    target_compile_definitions(AngrySanta_SIMD PRIVATE
        NOMINMAX             # prevent min/max macros from windows.h
    )
endif()

# --- Compiler flags per architecture -----------------------------------------
if(TARGET_ARCH STREQUAL "X86_64")
    target_compile_definitions(AngrySanta_SIMD PRIVATE ARCH_X86_64)

    if(MSVC)
        target_compile_options(AngrySanta_SIMD PRIVATE /arch:SSE2)
        target_compile_options(AngrySanta_SIMD PRIVATE /O2 /fp:fast) # O3 not exists in MSVC
    else()  # GCC / Clang on x86-64
        target_compile_options(AngrySanta_SIMD PRIVATE -msse2)
        target_compile_options(AngrySanta_SIMD PRIVATE -O3 -ffast-math)
    endif()

    target_compile_definitions(AngrySanta_SIMD PRIVATE HAS_SSE2)
    
    message(STATUS "SIMD: SSE2 enabled")
elseif(TARGET_ARCH STREQUAL "ARM64")
    target_compile_definitions(AngrySanta_SIMD PRIVATE ARCH_ARM64)

    if(NOT MSVC)
        target_compile_options(AngrySanta_SIMD PRIVATE -march=armv8-a)
        target_compile_definitions(AngrySanta_SIMD PRIVATE HAS_NEON)
        message(STATUS "SIMD: ARM NEON enabled (AArch64 baseline)")
        
        target_compile_options(AngrySanta_SIMD PRIVATE -O3 -ffast-math)
    endif()
endif()

# --- Release / Debug ---------------------------------------------------------
target_compile_definitions(AngrySanta_SIMD PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:DEBUG>
)

target_compile_definitions(AngrySanta_SIMD PRIVATE EXEPATH="${CMAKE_BINARY_DIR}/")
