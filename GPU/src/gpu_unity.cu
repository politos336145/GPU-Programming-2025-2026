// ============================================================================
// Manual Unity Build for CUDA sources
//
// Merges all .cu translation units into ONE so that __constant__ d_params
// (defined in helpers.cuh) is shared across every kernel.
//
// We don't use CMake's UNITY_BUILD property because it merges main.cu too,
// which triggers CUB header conflicts (dispatch_segmented_sort.cuh).
//
// main.cu is always compiled separately (it contains main() and no device
// kernels that read d_params).
// ============================================================================

#include "grid.cu"
#include "simulation.cu"
#include "snowpack.cu"
