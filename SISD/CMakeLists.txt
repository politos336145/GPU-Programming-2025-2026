cmake_minimum_required(VERSION 3.10)
project(AngrySanta_SISD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

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
add_executable(AngrySanta_SISD ${SOURCES} ${HEADERS})

target_include_directories(AngrySanta_SISD PRIVATE ${SRC_DIR} ${SHARED_SRC_DIR} ${SHARED_INCLUDE_DIR})

# --- Link against math library (required on Linux) ----------------------------
if(NOT MSVC)
    target_link_libraries(AngrySanta_SISD PRIVATE m)
endif()

# --- MSVC portability defines ------------------------------------------------
if(MSVC)
    target_compile_definitions(AngrySanta_SISD PRIVATE
        NOMINMAX             # prevent min/max macros from windows.h
    )
endif()

if (MSVC)
    target_compile_options(AngrySanta_SISD PRIVATE
        /Od # DISABLE ALL OPTIMIZATIONS
        /Ob0 # no inlining
        /fp:precise # disable fast math optimizations
    )
else()
    target_compile_options(AngrySanta_SISD PRIVATE
        -O0 # DISABLE ALL OPTIMIZATIONS
        -fno-inline # no inlining
        -fno-tree-vectorize # no auto-vectorization
        -fno-unroll-loops # no loop unrolling
    )
endif()

target_compile_definitions(AngrySanta_SISD PRIVATE EXEPATH="${CMAKE_BINARY_DIR}/")