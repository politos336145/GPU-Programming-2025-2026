// ============================================================================
// Spatial grid construction and utilities
// Provides functions to build a uniform grid for neighbor search, including
// hashing, sorting, and cell range construction.
// ============================================================================

#include "include/grid.cuh"
#include "include/helpers.cuh"
#include "include/kernel.cuh"
#include "include/simulation.cuh"
#include "../../shared/include/global.h"

#include <cub/cub.cuh>

// On CUDA 10.x (Jetson Nano), __ldg is not pulled into the include chain by cuda_runtime.h.
// Placed here - AFTER all system headers
#if (__CUDACC_VER_MAJOR__ < 11) && !defined(__ldg)
  #define __ldg(p) (*(p))
#endif

// ============================================================================
// Memory management
// ============================================================================
/**
 * @brief Compute grid dimensions from domain bounds and allocate device arrays.
 * 
 * @param grid           GridData struct to initialize.
 * @param shellCapacity  Maximum number of shell particles (capacity of per-particle arrays).
 * @param params         SimParams providing slope geometry, domainLength, and cellSize.
 */
void allocateGrid(GridData& grid, int shellCapacity, const SimParams& params) {
  computeGridDimensions(grid, params);

  // -----------------------------------------------------------------------
  // Compact hash table: instead of allocating cellStart/cellEnd for the full
  // numCells (~9.8M for default domain), use modular hashing to map the
  // spatial linear index into a much smaller table.
  // hashTableSize = nextPowerOf2(capacity * 8), clamped to [4096, numCells].
  // -----------------------------------------------------------------------
  {
    int target = shellCapacity * 8; // low load factor → minimal false positives
    int ht = 4096;
    while (ht < target) ht <<= 1;
    if (ht > grid.numCells) ht = grid.numCells; // never bigger than full grid
    grid.hashTableSize = ht;
    grid.hashTableMask = ht - 1;
  }

  printf("[Grid] dims = %d x %d x %d  (%d cells, cellSize = %.4f m)\n",
         grid.gridDimX, grid.gridDimY, grid.gridDimZ, grid.numCells, grid.cellSize);
  printf("[Grid] origin = (%.2f, %.2f, %.2f)\n", grid.originX, grid.originY, grid.originZ);
  printf("[Grid] hashTable = %d entries (%.1f KB) - compact modular hash\n",
         grid.hashTableSize, grid.hashTableSize * 2 * sizeof(int) / 1024.0f);

  // -----------------------------------------------------------------------
  // Allocate per-particle arrays (size = capacity)
  // -----------------------------------------------------------------------
  int cap = shellCapacity;
  CUDA_CHECK(cudaMalloc(&grid.cellHash,          cap * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&grid.particleIndex,     cap * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&grid.cellHashAlt,       cap * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&grid.particleIndexAlt,  cap * sizeof(uint32_t)));

  // Zero cellHashAlt so clearPrevCellsKernel reads valid (in-bounds) hashes
  // on the first frame after allocation (when no prior sort output exists).
  CUDA_CHECK(cudaMemset(grid.cellHashAlt, 0, cap * sizeof(uint32_t)));

  // -----------------------------------------------------------------------
  // Pre-allocate CUB radix sort temp storage
  // -----------------------------------------------------------------------
  cub::DoubleBuffer<uint32_t> d_keys(grid.cellHash, grid.cellHashAlt);
  cub::DoubleBuffer<uint32_t> d_vals(grid.particleIndex, grid.particleIndexAlt);

  grid.d_sortTemp = nullptr;
  grid.sortTempBytes = 0;

  cub::DeviceRadixSort::SortPairs(
    nullptr,
    grid.sortTempBytes,
    d_keys,
    d_vals,
    cap);

  CUDA_CHECK(cudaMalloc(&grid.d_sortTemp, grid.sortTempBytes));
  printf("[Grid] CUB radix sort temp: %.1f KB\n", grid.sortTempBytes / 1024.0f);

  // -----------------------------------------------------------------------
  // Allocate per-cell arrays (compact hash table)
  // -----------------------------------------------------------------------
  size_t cellBytes = grid.hashTableSize * sizeof(int);
  CUDA_CHECK(cudaMalloc(&grid.cellStart, cellBytes));
  CUDA_CHECK(cudaMalloc(&grid.cellEnd,   cellBytes));

  // Pre-initialise cell arrays so scatter-clear works from the first frame
  CUDA_CHECK(cudaMemset(grid.cellStart, INACTIVE, cellBytes));
  CUDA_CHECK(cudaMemset(grid.cellEnd,   INACTIVE, cellBytes));
}

/**
 * @brief Free all device memory owned by a GridData struct.
 * 
 * @param grid  GridData to deallocate (pointers set to nullptr, numCells to 0).
 */
void freeGrid(GridData& grid) {
  cudaFree(grid.cellHash);
  cudaFree(grid.particleIndex);
  cudaFree(grid.cellHashAlt);
  cudaFree(grid.particleIndexAlt);
  cudaFree(grid.d_sortTemp);
  cudaFree(grid.cellStart);
  cudaFree(grid.cellEnd);

  grid.cellHash          = nullptr;
  grid.particleIndex     = nullptr;
  grid.cellHashAlt       = nullptr;
  grid.particleIndexAlt  = nullptr;
  grid.d_sortTemp        = nullptr;
  grid.cellStart         = nullptr;
  grid.cellEnd           = nullptr;

  grid.numCells          = 0;
}

/**
 * @brief Resize the grid for a new shell capacity.
 * 
 * @param grid             GridData to resize.
 * @param newShellCapacity New maximum number of shell particles (capacity of per-particle arrays).
 * @param params           SimParams providing domain geometry and cell size.
 */
void resizeGrid(GridData& grid, int newShellCapacity, const SimParams& params) {
  freeGrid(grid);
  allocateGrid(grid, newShellCapacity, params);
}

// ============================================================================
// computeHashKernel - each particle computes its cell hash
// ============================================================================
/**
 * @brief Compute the spatial-hash cell index for each particle.
 * 
 * @param cellHash       Output cell hash per particle.
 * @param particleIndex  Output original particle index (identity map).
 * @param posX           Particle position X.
 * @param posY           Particle position Y.
 * @param posZ           Particle position Z.
 * @param originX        Grid origin X.
 * @param originY        Grid origin Y.
 * @param originZ        Grid origin Z.
 * @param cellSize       Uniform cell side length.
 * @param gridDimX       Grid dimension along X.
 * @param gridDimY       Grid dimension along Y.
 * @param gridDimZ       Grid dimension along Z.
 * @param hashTableMask  Mask for modular hashing into compact hash table.
 * @param N              Total number of particles.
 */
__global__ void computeHashKernel(
    uint32_t *__restrict__ cellHash,
    uint32_t *__restrict__ particleIndex,
    const float *__restrict__ posX, const float *__restrict__ posY, const float *__restrict__ posZ,
    float originX, float originY, float originZ,
    float cellSize,
    int gridDimX, int gridDimY, int gridDimZ,
    int hashTableMask,
    int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  // Compute cell coordinates
  int cx = __float2int_rd((__ldg(&posX[tid]) - originX) / cellSize);
  int cy = __float2int_rd((__ldg(&posY[tid]) - originY) / cellSize);
  int cz = __float2int_rd((__ldg(&posZ[tid]) - originZ) / cellSize);

  // Clamp to grid bounds
  cx = max(0, min(cx, gridDimX - 1));
  cy = max(0, min(cy, gridDimY - 1));
  cz = max(0, min(cz, gridDimZ - 1));

  // Spatial hash: prime-number mixing (see cellHash3D in helpers.cuh)
  uint32_t hash = cellHash3D(cx, cy, cz, hashTableMask);

  cellHash[tid]      = hash;
  particleIndex[tid] = (uint32_t)tid;
}

// ============================================================================
// clearPrevCellsKernel - scatter-clear only the cells used last frame
// Reads the PREVIOUS frame's sorted hash array to know which cells to reset.
// Runs on N threads; de-duplicates via boundary check (only first particle
// in each run of identical hashes writes the clear).
// ============================================================================
/**
 * @brief Clear cells from the previous frame.
 * 
 * @param prevSortedHash  Previous frame's sorted hash array.
 * @param cellStart       Output start index for each cell (inclusive).
 * @param cellEnd         Output end index for each cell (exclusive).
 * @param shellN          Total number of shell particles.
 */
__global__ void clearPrevCellsKernel(
    const uint32_t *__restrict__ prevSortedHash,
    int *__restrict__ cellStart, int *__restrict__ cellEnd,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shellN) return;

  // Clear only unique cells: first occurrence of each hash value
  uint32_t hash = __ldg(&prevSortedHash[tid]);
  if (tid == 0 || hash != __ldg(&prevSortedHash[tid - 1])) {
    cellStart[hash] = INACTIVE;
    cellEnd[hash]   = INACTIVE;
  }
}

// ============================================================================
// buildCellRangesKernel - find start/end of each cell in the sorted array
// ============================================================================
/**
 * @brief Find start/end indices of each cell in the sorted hash array.
 *
 * @param cellHash   Sorted cell hash array.
 * @param cellStart  Output start index for each cell (inclusive).
 * @param cellEnd    Output end index for each cell (exclusive).
 * @param shellN     Number of shell particles.
 */
__global__ void buildCellRangesKernel(
    const uint32_t *__restrict__ cellHash,
    int *__restrict__ cellStart, int *__restrict__ cellEnd,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shellN) return;

  // First particle or hash differs from predecessor → new cell starts here
  uint32_t hash = __ldg(&cellHash[tid]);
  if (tid == 0 || hash != __ldg(&cellHash[tid - 1]))
    cellStart[hash] = tid;

  // Last particle or hash differs from successor → cell ends here
  if (tid == shellN - 1 || hash != __ldg(&cellHash[tid + 1]))
    cellEnd[hash] = tid + 1; // exclusive end
}

// ============================================================================
// launchBuildGrid - full pipeline: hash → sort → cellRanges
// ============================================================================
/**
 * @brief Build the spatial grid: hash → sort → cell ranges.
 * 
 * @param grid    GridData with allocated per-particle and per-cell arrays.
 * @param ps      ParticleSystem providing particle positions.
 * @param shellN  Number of shell particles.
 * @param stream  CUDA stream for async execution.
 */
void launchBuildGrid(GridData& grid, const ParticleSystem& ps, int shellN, cudaStream_t stream) {

  // -----------------------------------------------------------------------
  // Step 1: sparse-clear cells occupied in the PREVIOUS frame.
  // cellHashAlt still holds the previous frame's sorted hashes (CUB output) so we read them here BEFORE the new sort overwrites them
  // -----------------------------------------------------------------------
  const int threadsPerBlock1 = 256;
  clearPrevCellsKernel<<<gridSize(shellN, threadsPerBlock1), threadsPerBlock1, 0, stream>>>(grid.cellHashAlt, grid.cellStart, grid.cellEnd, shellN);
  CUDA_CHECK(cudaGetLastError());

  // -----------------------------------------------------------------------
  // Step 2: compute hash for each particle (with modular hash)
  // -----------------------------------------------------------------------
  const int threadsPerBlock2 = 256;
  computeHashKernel<<<gridSize(shellN, threadsPerBlock2), threadsPerBlock2, 0, stream>>>(
    grid.cellHash, grid.particleIndex,
    ps.posX, ps.posY, ps.posZ,
    grid.originX, grid.originY, grid.originZ,
    grid.cellSize,
    grid.gridDimX, grid.gridDimY, grid.gridDimZ,
    grid.hashTableMask,
    shellN);  
  CUDA_CHECK(cudaGetLastError());

  // -----------------------------------------------------------------------
  // Step 3: CUB radix sort (cellHash, particleIndex) by hash
  // -----------------------------------------------------------------------
  // Compute minimum bits needed - uses hashTableSize (compact) instead of numCells
  int sortBits = 0;
  { 
    unsigned v = (unsigned)(grid.hashTableSize > 0 ? grid.hashTableSize - 1 : 0);
    while (v) { sortBits++; v >>= 1; }
  }
  sortBits = ((sortBits + 3) / 4) * 4; // round to 4-bit radix boundary
  if (sortBits < 4)  sortBits = 4;
  if (sortBits > 32) sortBits = 32;

  cub::DoubleBuffer<uint32_t> d_keys(grid.cellHash, grid.cellHashAlt);
  cub::DoubleBuffer<uint32_t> d_vals(grid.particleIndex, grid.particleIndexAlt);
  cudaError_t sortErr = cub::DeviceRadixSort::SortPairs(
    grid.d_sortTemp,
    grid.sortTempBytes,
    d_keys,
    d_vals,
    shellN,
    0, sortBits,  // begin_bit, end_bit (only needed bits)
    stream);      // explicit stream for capture-safety
  CUDA_CHECK(sortErr);
  CUDA_CHECK(cudaGetLastError());
  uint32_t* sortedHashes = d_keys.Current();
  uint32_t* sortedIndices = d_vals.Current();

  // -----------------------------------------------------------------------
  // Step 4: build new cell ranges from the freshly-sorted hashes.
  // Uses CUB output arrays as the canonical sorted representation to avoid extra D2D copies per frame.
  // -----------------------------------------------------------------------
  const int threadsPerBlock3 = 256;
  buildCellRangesKernel<<<gridSize(shellN, threadsPerBlock3), threadsPerBlock3, 0, stream>>>(sortedHashes, grid.cellStart, grid.cellEnd, shellN);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// printGridStats - transfer cellStart/cellEnd to CPU and compute stats
// ============================================================================
/**
 * @brief Print grid occupancy statistics to stdout (CPU-side, transfers data from device).
 * 
 * @param grid   GridData to inspect.
 * @param shellN Number of shell particles.
 * @param frame  Current simulation frame number (for log labelling).
 */
void printGridStats(const GridData& grid, int shellN, int frame) {
  int numCells = grid.hashTableSize;

  int* h_start = new int[numCells];
  int* h_end   = new int[numCells];
  CUDA_CHECK(cudaMemcpy(h_start, grid.cellStart, numCells * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_end,   grid.cellEnd,   numCells * sizeof(int), cudaMemcpyDeviceToHost));

  int nonEmpty = 0;
  int maxBucket = 0;
  long long totalCount = 0;

  for (int c = 0; c < numCells; c++) {
    if (h_start[c] == INACTIVE) continue; // empty cell

    int count = h_end[c] - h_start[c];
    totalCount += count;
    nonEmpty++;
    if (count > maxBucket) maxBucket = count;
  }

  float avgBucket = (nonEmpty > 0) ? (float)totalCount / nonEmpty : 0.0f;
  printf("[Grid] Frame %d: non-empty = %d / %d, max bucket = %d, avg = %.1f, sum = %lld (expected %d)\n", frame, nonEmpty, numCells, maxBucket, avgBucket, totalCount, shellN);

  delete[] h_start;
  delete[] h_end;
}