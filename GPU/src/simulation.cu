#include "include/helpers.cuh"
#include "include/kernel.cuh"
#include "include/simulation.cuh"

// On CUDA 10.x (Jetson Nano), __ldg is not pulled into the include chain by cuda_runtime.h.
// Placed here - AFTER all system headers
#if (__CUDACC_VER_MAJOR__ < 11) && !defined(__ldg)
  #define __ldg(p) (*(p))
#endif

// anonymous namespace for helper functions and constants
namespace SimulationNS {
  const int threshold = 256;
  
  // Block size for the neighbor collision kernel.
  // __launch_bounds__(512, 3) on the kernel guarantees the compiler targets
  // 3 resident blocks of 512 threads → ~21 regs/thread.  Shared memory per
  // block is ~12.4 KB (6 arrays × 16 warps × 33 slots), fitting 3 blocks
  // in 48 KB → 1536 threads → 75% occupancy on SM 5.x.
  // No special-casing for Jetson Nano: the launch_bounds already ensure
  // correct resource usage.  Launching with fewer threads wastes shared
  // memory (sized for 16 warps regardless) and tanks occupancy.
  int collisionBlockSizeForCurrentGpu() {
    int blockSize = 512;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
        if (blockSize > prop.maxThreadsPerBlock) blockSize = prop.maxThreadsPerBlock;
    }

    return blockSize;
  }
}

// ============================================================================
// copyParamsToGPU - upload SimParams to GPU constant memory (d_params)
// ============================================================================
/**
 * @brief Copy SimParams to GPU constant memory (d_params).
 * 
 * @param params  Host-side SimParams to upload.
 */
void copyParamsToGPU(const SimParams &params) {
  CUDA_CHECK(cudaMemcpyToSymbol(d_params, &params, sizeof(SimParams), 0, cudaMemcpyHostToDevice));
}

// ============================================================================
// Memory management
// ============================================================================

/**
 * @brief Allocate device memory for all SoA arrays of a ParticleSystem.
 * 
 * @param ps        ParticleSystem whose device pointers will be allocated.
 * @param capacity  Number of particles the system can hold.
 */
void allocateParticleSystem(ParticleSystem &ps, int capacity) {
  ps.capacity = capacity;

  CUDA_CHECK(cudaMalloc(&ps.posX, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.posY, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.posZ, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.velX, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.velY, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.velZ, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.forceX, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.forceY, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.forceZ, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.mass, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.radius, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.wetness, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.attachLocalX, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.attachLocalY, capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ps.attachLocalZ, capacity * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&ps.state, capacity * sizeof(ShellParticleState)));
  CUDA_CHECK(cudaMemset(ps.state, INACTIVE, capacity * sizeof(ShellParticleState)));
}

/**
 * @brief Free all device memory owned by a ParticleSystem.
 * 
 * @param ps  ParticleSystem to deallocate (capacity set to 0).
 */
void freeParticleSystem(ParticleSystem &ps) {
  cudaFree(ps.posX);
  cudaFree(ps.posY);
  cudaFree(ps.posZ);
  cudaFree(ps.velX);
  cudaFree(ps.velY);
  cudaFree(ps.velZ);
  cudaFree(ps.forceX);
  cudaFree(ps.forceY);
  cudaFree(ps.forceZ);
  cudaFree(ps.mass);
  cudaFree(ps.radius);
  cudaFree(ps.state);
  cudaFree(ps.wetness);
  cudaFree(ps.attachLocalX);
  cudaFree(ps.attachLocalY);
  cudaFree(ps.attachLocalZ);

  ps.capacity = 0;
}

// ============================================================================
// growShell - grow shell to larger capacity, preserving active particles
// ============================================================================
/**
 * @brief Grow the shell to a larger capacity, preserving active particles.
 * 
 * @param shell        Shell ParticleSystem to grow (modified in-place).
 * @param shellN       Number of active shell particles to preserve.
 * @param newCapacity  New capacity (must be > shell.capacity).
 */
void growShell(ParticleSystem &shell, int shellN, int newCapacity) {
  ParticleSystem old = shell;
  
  allocateParticleSystem(shell, newCapacity);
  
  if (shellN > 0) {
    size_t fb = shellN * sizeof(float);
    size_t sb = shellN * sizeof(ShellParticleState);
  
    CUDA_CHECK(cudaMemcpy(shell.posX,   old.posX,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.posY,   old.posY,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.posZ,   old.posZ,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.velX,   old.velX,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.velY,   old.velY,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.velZ,   old.velZ,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.forceX, old.forceX, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.forceY, old.forceY, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.forceZ, old.forceZ, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.mass,   old.mass,   fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.radius, old.radius, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.wetness,old.wetness, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.attachLocalX, old.attachLocalX, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.attachLocalY, old.attachLocalY, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.attachLocalZ, old.attachLocalZ, fb, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(shell.state,  old.state,  sb, cudaMemcpyDeviceToDevice));
  }
  
  freeParticleSystem(old);
}

// ============================================================================
// neighborCollisionKernel - soft-sphere (spring-dashpot) + cohesion
// ============================================================================
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_WARPS = 16;  // 512 / 32

/**
 * @brief Soft-sphere (spring-dashpot) + cohesion collision using the spatial grid.
 *        Traverses 27 neighbouring cells.  Non-symmetric: each thread computes
 *        the net force on its own particle from ALL neighbours (j != i).
 *        Forces accumulate entirely in registers — no atomicAdd anywhere.
 *        Thread i is the sole writer to forceX/Y/Z[i], so a regular += suffices
 *        (tether kernel already wrote its contribution before this kernel runs
 *        on the same stream).
 *
 *        Cell particles are loaded into per-warp shared-memory tiles (32+1 slots
 *        per warp, +1 padding against bank conflicts).  Velocity is loaded
 *        on-demand from global memory (via __ldg) only for actual contacts
 *        (dist < sumR), keeping shared-memory footprint small enough for
 *        3 resident blocks per SM on SM 5.0+.
 * 
 * @param posX           Particle position X.
 * @param posY           Particle position Y.
 * @param posZ           Particle position Z.
 * @param velX           Particle velocity X.
 * @param velY           Particle velocity Y.
 * @param velZ           Particle velocity Z.
 * @param forceX         Force accumulation X (single-writer per thread, regular +=).
 * @param forceY         Force accumulation Y.
 * @param forceZ         Force accumulation Z.
 * @param radius         Per-particle radius.
 * @param mass           Per-particle mass.
 * @param state          Per-particle state (only ACTIVE particles processed).
 * @param particleIndex  Sorted particle indices (from grid build).
 * @param cellStart      Start index of each cell in the sorted array.
 * @param cellEnd        Exclusive end index of each cell.
 * @param gridDimX       Grid dimension along X.
 * @param gridDimY       Grid dimension along Y.
 * @param gridDimZ       Grid dimension along Z.
 * @param originX        Grid origin X.
 * @param originY        Grid origin Y.
 * @param originZ        Grid origin Z.
 * @param cellSize       Uniform cell side length.
 * @param hashTableMask  Mask for modular hashing into compact hash table.
 * @param shellN         Number of shell particles.
 */
// TODO da capire bene cosa fa
__global__ __launch_bounds__(512, 3)
void neighborCollisionKernel(
    const float *__restrict__ posX, const float *__restrict__ posY, const float *__restrict__ posZ,
    const float *__restrict__ velX, const float *__restrict__ velY, const float *__restrict__ velZ,
    float *__restrict__ forceX, float *__restrict__ forceY, float *__restrict__ forceZ,
    const float *__restrict__ radius,
    const float *__restrict__ mass,
    const ShellParticleState *__restrict__ state,
    const uint32_t *__restrict__ particleIndex,
    const int *__restrict__ cellStart, const int *__restrict__ cellEnd,
    int gridDimX, int gridDimY, int gridDimZ,
    float originX, float originY, float originZ,
    float cellSize,
    int hashTableMask,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shellN) return;
  if (__ldg(reinterpret_cast<const int*>(&state[tid])) != ACTIVE) return;
  
  // Per-warp shared memory tiles: each warp has its own 32+1 slots
  // (+1 padding avoids potential bank conflicts across rows).
  
  __shared__ float s_px[MAX_WARPS][WARP_SIZE + 1], s_py[MAX_WARPS][WARP_SIZE + 1], s_pz[MAX_WARPS][WARP_SIZE + 1];
  __shared__ float s_r[MAX_WARPS][WARP_SIZE + 1];
  __shared__ int   s_idx[MAX_WARPS][WARP_SIZE + 1];
  __shared__ int   s_state[MAX_WARPS][WARP_SIZE + 1];  
  // [DISABLED] Velocity in shared tile: loading velX/Y/Z into shared memory
  // eliminates random __ldg reads in the collision branch, but adds 3 float
  // arrays × 16 warps × 33 slots = 6336B → total smem jumps from ~13KB to ~19KB.
  // On SM 5.0 (48KB shared per SM) this drops occupancy from 3 to 2 resident
  // blocks, causing a ~25% throughput loss at large N.  Re-enable on GPUs with
  // ≥96KB shared memory (SM 8.0+ / Ampere) by uncommenting these lines and
  // changing __launch_bounds__ from (512,3) to (512,2).
  // __shared__ float s_vx[MAX_WARPS][WARP_SIZE + 1], s_vy[MAX_WARPS][WARP_SIZE + 1], s_vz[MAX_WARPS][WARP_SIZE + 1];
  
  int warpId = threadIdx.x >> 5;
  int lane   = threadIdx.x & 31;

  float pi_x = __ldg(&posX[tid]), pi_y = __ldg(&posY[tid]), pi_z = __ldg(&posZ[tid]);
  float vi_x = __ldg(&velX[tid]), vi_y = __ldg(&velY[tid]), vi_z = __ldg(&velZ[tid]);

  float ri   = __ldg(&radius[tid]);

  int cx = __float2int_rd((pi_x - originX) / cellSize);
  int cy = __float2int_rd((pi_y - originY) / cellSize);
  int cz = __float2int_rd((pi_z - originZ) / cellSize);
  cx = max(0, min(cx, gridDimX - 1));
  cy = max(0, min(cy, gridDimY - 1));
  cz = max(0, min(cz, gridDimZ - 1));

  float k_pen   = d_params.stiffness;
  float k_damp  = d_params.collisionDamping;
  float k_coh   = d_params.cohesion;
  float r_cut   = d_params.cohesionRadius;
  float mu_p    = d_params.particleFriction;

  float fi_x = 0.0f, fi_y = 0.0f, fi_z = 0.0f;

  // Traverse 27 neighbor cells
  for (int dz = -1; dz <= 1; dz++) {
    int nz = cz + dz;
    if (nz < 0 || nz >= gridDimZ) continue;

    for (int dy = -1; dy <= 1; dy++) {
      int ny = cy + dy;
      if (ny < 0 || ny >= gridDimY) continue;

      for (int dx = -1; dx <= 1; dx++) {
        int nx_cell = cx + dx;
        if (nx_cell < 0 || nx_cell >= gridDimX) continue;

        int hash = (int)cellHash3D(nx_cell, ny, nz, hashTableMask);
        int start = __ldg(&cellStart[hash]);
        if (start == INACTIVE) continue;

        int end = __ldg(&cellEnd[hash]);
        int cellCount = end - start;

        // Process cell particles in warp-cooperative tiles of WARP_SIZE
        for (int tileBase = 0; tileBase < cellCount; tileBase += WARP_SIZE) {
          int tileEnd = min(tileBase + WARP_SIZE, cellCount);
          int tileLen = tileEnd - tileBase;

          // Each lane loads one particle into its warp's shared memory
          if (lane < tileLen) {
            int sortedIdx = start + tileBase + lane;
            int j = __ldg(&particleIndex[sortedIdx]);
            if (j < 0 || j >= shellN) {
              s_idx[warpId][lane] = -1;
              s_state[warpId][lane] = -1;
            }
            else {
              s_idx[warpId][lane]   = j;
              s_px[warpId][lane]    = __ldg(&posX[j]);
              s_py[warpId][lane]    = __ldg(&posY[j]);
              s_pz[warpId][lane]    = __ldg(&posZ[j]);
              s_r[warpId][lane]     = __ldg(&radius[j]);
              s_state[warpId][lane] = __ldg(reinterpret_cast<const int*>(&state[j]));
              // [DISABLED] Velocity in shared tile (see note at declaration above)
              // s_vx[warpId][lane] = __ldg(&velX[j]);
              // s_vy[warpId][lane] = __ldg(&velY[j]);
              // s_vz[warpId][lane] = __ldg(&velZ[j]);
            }
          }

          __syncwarp();

          for (int t = 0; t < tileLen; t++) {
            int j = s_idx[warpId][t];
            if (j < 0 || j >= shellN) continue;
            if (s_state[warpId][t] != ACTIVE) continue;
            if (j == tid) continue;  // skip self
          
            float dx_ij = s_px[warpId][t] - pi_x;
            float dy_ij = s_py[warpId][t] - pi_y;
            float dz_ij = s_pz[warpId][t] - pi_z;
            float dist2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

            float rj = s_r[warpId][t];
            float sumR = ri + rj;
            float cutoff = (k_coh > 0.0f && r_cut > sumR) ? r_cut : sumR;
            if (dist2 >= cutoff * cutoff || dist2 < 1e-12f) continue;
            
            float invDist = rsqrtf(dist2);
            float dist = dist2 * invDist;
            float n_x = dx_ij * invDist;
            float n_y = dy_ij * invDist;
            float n_z = dz_ij * invDist;

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            if (dist < sumR) {
              float pen = sumR - dist;
              // TODO [DISABLED alternative] If velocity-in-shared is enabled, replace with:
              // float vrel_x = s_vx[warpId][t] - vi_x;
              // float vrel_y = s_vy[warpId][t] - vi_y;
              // float vrel_z = s_vz[warpId][t] - vi_z;
              // Velocity loaded on-demand via __ldg (contacts are rare vs distance checks).
              float vrel_x = __ldg(&velX[j]) - vi_x;
              float vrel_y = __ldg(&velY[j]) - vi_y;
              float vrel_z = __ldg(&velZ[j]) - vi_z;
              float vn = vrel_x * n_x + vrel_y * n_y + vrel_z * n_z;

              float f_spring = k_pen * pen;
              float f_damp   = k_damp * vn;
              float f_mag    = fmaxf(f_spring - f_damp, 0.0f);

              fx = f_mag * n_x;
              fy = f_mag * n_y;
              fz = f_mag * n_z;

              // Tangential friction (Coulomb model between particles)
              if (mu_p > 0.0f) {
                float vt_x = vrel_x - vn * n_x;
                float vt_y = vrel_y - vn * n_y;
                float vt_z = vrel_z - vn * n_z;
                float vt2 = vt_x * vt_x + vt_y * vt_y + vt_z * vt_z;
                if (vt2 > 1e-16f) {
                  float invVt = rsqrtf(vt2);
                  float fFric = mu_p * fabsf(f_mag);
                  fx -= fFric * vt_x * invVt;
                  fy -= fFric * vt_y * invVt;
                  fz -= fFric * vt_z * invVt;
                }
              }
            } else if (k_coh > 0.0f) {
              float coh_mag = k_coh * (1.0f - dist / r_cut);
              fx = -coh_mag * n_x;
              fy = -coh_mag * n_y;
              fz = -coh_mag * n_z;
            }

            fi_x += fx;
            fi_y += fy;
            fi_z += fz;
          }

          __syncwarp(); // ensure tile is consumed before next load
        }
      }
    }
  }

  // Thread tid is the sole writer to forceX/Y/Z[tid] → regular += (no atomicAdd)
  forceX[tid] += fi_x;
  forceY[tid] += fi_y;
  forceZ[tid] += fi_z;
}

/**
 * @brief Launch the neighbor collision kernel using the spatial grid.
 *        Non-symmetric: each thread handles all neighbours (j != tid),
 *        accumulates in registers, single regular store per component.
 *        Block size 512.
 * 
 * @param ps      ParticleSystem (reads pos/vel/radius, writes force*).
 * @param grid    GridData with sorted cell hash tables.
 * @param shellN  Number of shell particles.
 * @param stream  CUDA stream for async execution.
 */
void launchNeighborCollision(ParticleSystem &ps, const GridData &grid, int shellN, cudaStream_t stream) {
  if (shellN <= 0) return;

  const int threadsPerBlock = SimulationNS::collisionBlockSizeForCurrentGpu();
  neighborCollisionKernel<<<gridSize(shellN, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    ps.posX, ps.posY, ps.posZ,
    ps.velX, ps.velY, ps.velZ,
    ps.forceX, ps.forceY, ps.forceZ,
    ps.radius,
    ps.mass,
    ps.state,
    grid.particleIndexAlt,
    grid.cellStart, grid.cellEnd,
    grid.gridDimX, grid.gridDimY, grid.gridDimZ,
    grid.originX, grid.originY, grid.originZ,
    grid.cellSize,
    grid.hashTableMask,
    shellN);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// bruteForceCollisionKernel - O(N²) direct pair interaction
//
// For small shell counts (≤ BF_COLLISION_THRESHOLD), the overhead of building
// the spatial grid (5-8 GPU commands: clear + hash + CUB sort + ranges +
// collision) exceeds the benefit.  This kernel performs a direct O(N²) scan
// in a single launch with zero infrastructure overhead.
//
// Each thread i computes forces from ALL j ≠ i.  No atomicAdd needed: each
// thread writes only to its own force index.  Forces are accumulated on top
// of the tether forces already present in forceX/Y/Z (written by the
// preceding forceTetherKernel on the same stream).
// ============================================================================

/**
 * @brief Brute-force O(N²) collision for small shell counts (≤ BF_COLLISION_THRESHOLD).
 *        Each thread computes forces from ALL j ≠ i.  No atomicAdd needed.
 *        Block size 256, launch bounds ensure correct resource usage on all GPUs.
 *        Velocity is loaded on-demand from global memory only for actual contacts
 *        (dist < sumR), keeping register usage low enough for 3 resident blocks on SM 5.0+.
 *
 * @param posX   Particle position X.
 * @param posY   Particle position Y.
 * @param posZ   Particle position Z.
 * @param velX   Particle velocity X.
 * @param velY   Particle velocity Y.
 * @param velZ   Particle velocity Z.
 * @param forceX Force accumulation X (single-writer per thread, regular +=).
 * @param forceY Force accumulation Y.
 * @param forceZ Force accumulation Z.
 * @param radius Per-particle radius.
 * @param state  Per-particle state (only ACTIVE particles processed).
 * @param shellN Number of shell particles.
 * 
 * @note This kernel is not spatially coherent and has O(N²) complexity, so it should only be used for small N (≤ BF_COLLISION_THRESHOLD).
 *       For larger N, use neighborCollisionKernel with the spatial grid for O(N) complexity and better memory access patterns. 
 */
// TODO da capire bene cosa fa
__global__ __launch_bounds__(256, 5)
void bruteForceCollisionKernel(
    const float *__restrict__ posX, const float *__restrict__ posY, const float *__restrict__ posZ,
    const float *__restrict__ velX, const float *__restrict__ velY, const float *__restrict__ velZ,
    float *__restrict__ forceX, float *__restrict__ forceY, float *__restrict__ forceZ,
    const float *__restrict__ radius,
    const ShellParticleState *__restrict__ state,
    int shellN)
{
  // TODO nessuna condizione di uscita anticipata?
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (tid < shellN) && (__ldg(reinterpret_cast<const int*>(&state[tid])) == ACTIVE);

  // ---- Shared memory for N-body collision tiling (+1 padding) ----
  __shared__ float sh_px[SimulationNS::threshold +1], sh_py[SimulationNS::threshold +1], sh_pz[SimulationNS::threshold +1];
  __shared__ float sh_vx[SimulationNS::threshold +1], sh_vy[SimulationNS::threshold +1], sh_vz[SimulationNS::threshold +1];
  __shared__ float sh_r[SimulationNS::threshold +1];
  __shared__ int   sh_st[SimulationNS::threshold +1];
  
  float pi_x = 0.0f, pi_y = 0.0f, pi_z = 0.0f;
  float vi_x = 0.0f, vi_y = 0.0f, vi_z = 0.0f;
  float ri = 0.0f;

  if (active) {
    pi_x = __ldg(&posX[tid]); 
    pi_y = __ldg(&posY[tid]);
    pi_z = __ldg(&posZ[tid]);
    vi_x = __ldg(&velX[tid]);
    vi_y = __ldg(&velY[tid]);
    vi_z = __ldg(&velZ[tid]);
    ri   = __ldg(&radius[tid]);
  }

  float k_pen  = d_params.stiffness;
  float k_damp = d_params.collisionDamping;
  float k_coh  = d_params.cohesion;
  float r_cut  = d_params.cohesionRadius;
  float mu_p   = d_params.particleFriction;

  float fi_x = 0.0f, fi_y = 0.0f, fi_z = 0.0f;

  for (int tile = 0; tile < shellN; tile += SimulationNS::threshold) {
    int loadIdx = tile + threadIdx.x;
    if (loadIdx < shellN) {
      sh_px[threadIdx.x] = posX[loadIdx];
      sh_py[threadIdx.x] = posY[loadIdx];
      sh_pz[threadIdx.x] = posZ[loadIdx];
      sh_vx[threadIdx.x] = velX[loadIdx];
      sh_vy[threadIdx.x] = velY[loadIdx];
      sh_vz[threadIdx.x] = velZ[loadIdx];
      sh_r[threadIdx.x]  = radius[loadIdx];
      sh_st[threadIdx.x] = reinterpret_cast<const int*>(state)[loadIdx];
    } else
      sh_st[threadIdx.x] = -1;

    __syncthreads();

    if (active) {
      int tileEnd = min(SimulationNS::threshold, shellN - tile);
      for (int k = 0; k < tileEnd; k++) {
        int j = tile + k;
        if (j == tid) continue;
        if (sh_st[k] != ACTIVE) continue;

        float dx_ij = sh_px[k] - pi_x;
        float dy_ij = sh_py[k] - pi_y;
        float dz_ij = sh_pz[k] - pi_z;
        float dist2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

        float rj = sh_r[k];
        float sumR = ri + rj;
        float cutoff = (k_coh > 0.0f && r_cut > sumR) ? r_cut : sumR;

        if (dist2 >= cutoff * cutoff || dist2 < 1e-12f) continue;

        float invDist = rsqrtf(dist2);
        float dist = dist2 * invDist;
        float n_x = dx_ij * invDist;
        float n_y = dy_ij * invDist;
        float n_z = dz_ij * invDist;

        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        if (dist < sumR) {
          float pen = sumR - dist;
          float vrel_x = sh_vx[k] - vi_x;
          float vrel_y = sh_vy[k] - vi_y;
          float vrel_z = sh_vz[k] - vi_z;
          float vn = vrel_x * n_x + vrel_y * n_y + vrel_z * n_z;

          float f_spring = k_pen * pen;
          float f_damp   = k_damp * vn;
          float f_mag    = fmaxf(f_spring - f_damp, 0.0f);

          fx = f_mag * n_x;
          fy = f_mag * n_y;
          fz = f_mag * n_z;

          if (mu_p > 0.0f) {
            float vt_x = vrel_x - vn * n_x;
            float vt_y = vrel_y - vn * n_y;
            float vt_z = vrel_z - vn * n_z;
            float vt2 = vt_x * vt_x + vt_y * vt_y + vt_z * vt_z;
            if (vt2 > 1e-16f) {
              float invVt = rsqrtf(vt2);
              float fFric = mu_p * fabsf(f_mag);
              fx -= fFric * vt_x * invVt;
              fy -= fFric * vt_y * invVt;
              fz -= fFric * vt_z * invVt;
            }
          }
        } else if (k_coh > 0.0f) {
          float coh_mag = k_coh * (1.0f - dist / r_cut);
          fx = -coh_mag * n_x;
          fy = -coh_mag * n_y;
          fz = -coh_mag * n_z;
        }

        fi_x += fx;
        fi_y += fy;
        fi_z += fz;
      }
    }
    __syncthreads();
  }

  if (!active) return;

  // Accumulate on top of tether forces (no atomicAdd needed: unique i per thread)
  forceX[tid] += fi_x;
  forceY[tid] += fi_y;
  forceZ[tid] += fi_z;
}

/**
 * @brief Launch the brute-force collision kernel for small shell counts (≤ BF_COLLISION_THRESHOLD).
 *        Each thread computes forces from ALL j ≠ i.  No atomicAdd needed.
 *        Block size 256, launch bounds ensure correct resource usage on all GPUs.
 *        Velocity is loaded on-demand from global memory only for actual contacts
 *        (dist < sumR), keeping register usage low enough for 3 resident blocks on SM 5.0+.
 *
 * @param ps      ParticleSystem (reads pos/vel/radius, writes force*).
 * @param shellN  Number of shell particles.
 * @param stream  CUDA stream for async execution.
 */
void launchBruteForceCollision(ParticleSystem &ps, int shellN, cudaStream_t stream) {
  if (shellN <= 0) return;

  const int threadsPerBlock = 256;
  bruteForceCollisionKernel<<<gridSize(shellN, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    ps.posX, ps.posY, ps.posZ,
    ps.velX, ps.velY, ps.velZ,
    ps.forceX, ps.forceY, ps.forceZ,
    ps.radius,
    ps.state,
    shellN);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// shellPhysicsFusedKernel - ALL shell physics in ONE kernel launch
//
// Replaces forceTether + bruteForceCollision + integrateGround for the
// small-N path.  Eliminates 2 kernel launches and all intermediate global
// memory traffic for forces (everything stays in registers).
// ============================================================================
/**
 * @brief Fused kernel for ALL shell physics in ONE launch, for small N (≤ BF_COLLISION_THRESHOLD).
 *        Replaces forceTether + bruteForceCollision + integrateGround for the small-N path
 *        Eliminates 2 kernel launches and all intermediate global memory traffic for forces (everything stays in registers).
 *        Block size 256, launch bounds ensure correct resource usage on all GPUs.
 *        Velocity is loaded on-demand from global memory only for actual contacts (dist < sumR), keeping register usage low enough for 3 resident blocks on SM 5.0+.
 *
 * @param posX           Particle position X.
 * @param posY           Particle position Y.
 * @param posZ           Particle position Z.
 * @param velX           Particle velocity X.
 * @param velY           Particle velocity Y.
 * @param velZ           Particle velocity Z.
 * @param mass           Per-particle mass.
 * @param radiusArr      Per-particle radius.
 * @param state          Per-particle state (only ACTIVE particles processed).
 * @param attachLocalX   Local attachment point X for tethering.
 * @param attachLocalY   Local attachment point Y for tethering.
 * @param attachLocalZ   Local attachment point Z for tethering.
 * @param bpx            Ball position X (for tethering).
 * @param bpy            Ball position Y (for tethering).
 * @param bpz            Ball position Z (for tethering).
 * @param bvx            Ball velocity X (for tethering).
 * @param bvy            Ball velocity Y (for tethering).
 * @param bvz            Ball velocity Z (for tethering).
 * @param box            Ball angular velocity X (for tethering).
 * @param boy            Ball angular velocity Y (for tethering).
 * @param boz            Ball angular velocity Z (for tethering).
 * @param bqw            Ball quaternion W (for tethering).
 * @param bqx            Ball quaternion X (for tethering).
 * @param bqy            Ball quaternion Y (for tethering).
 * @param bqz            Ball quaternion Z (for tethering).
 * @param ballRadius     Ball radius (for tethering).
 * @param shellN         Number of shell particles.
 */
__global__ __launch_bounds__(256, 5)
void shellPhysicsFusedKernel(
    float *__restrict__ posX,  float *__restrict__ posY,  float *__restrict__ posZ,
    float *__restrict__ velX,  float *__restrict__ velY,  float *__restrict__ velZ,
    const float *__restrict__ mass,
    const float *__restrict__ radiusArr,
    const ShellParticleState *__restrict__ state,
    const float *__restrict__ attachLocalX, const float *__restrict__ attachLocalY, const float *__restrict__ attachLocalZ,
    float bpx, float bpy, float bpz,
    float bvx, float bvy, float bvz,
    float box, float boy, float boz,
    float bqw, float bqx, float bqy, float bqz,
    float ballRadius,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (tid < shellN) && (__ldg(reinterpret_cast<const int*>(&state[tid])) == ACTIVE);

  // ---- Shared memory for N-body collision tiling (+1 padding) ----
  __shared__ float sh_px[SimulationNS::threshold + 1], sh_py[SimulationNS::threshold + 1], sh_pz[SimulationNS::threshold + 1];
  __shared__ float sh_vx[SimulationNS::threshold + 1], sh_vy[SimulationNS::threshold + 1], sh_vz[SimulationNS::threshold + 1];
  __shared__ float sh_r[SimulationNS::threshold + 1];
  __shared__ int   sh_st[SimulationNS::threshold + 1];

  float m = 0.0f, ri = 0.0f;
  float pi_x = 0.0f, pi_y = 0.0f, pi_z = 0.0f;
  float vi_x = 0.0f, vi_y = 0.0f, vi_z = 0.0f;
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;

  if (active) {
    m    = __ldg(&mass[tid]);
    ri   = __ldg(&radiusArr[tid]);
    pi_x = posX[tid]; pi_y = posY[tid]; pi_z = posZ[tid];
    vi_x = velX[tid]; vi_y = velY[tid]; vi_z = velZ[tid];

    // ---- Gravity + linear damping ----
    float kd = d_params.damping;
    fx = -kd * m * vi_x;
    fy = -m * d_params.gravity - kd * m * vi_y;
    fz = -kd * m * vi_z;

    // ---- Tether spring-damper ----
    float K = d_params.shellTetherK;
    float D = d_params.shellTetherDamp;
    float lx = __ldg(&attachLocalX[tid]);
    float ly = __ldg(&attachLocalY[tid]);
    float lz = __ldg(&attachLocalZ[tid]);
    float wx, wy, wz;
    quatRotateVec(bqw, bqx, bqy, bqz, lx, ly, lz, wx, wy, wz);
    float tDist = ballRadius + ri;
    float tpx = bpx + wx * tDist, tpy = bpy + wy * tDist, tpz = bpz + wz * tDist;
    float rrx = wx * tDist, rry = wy * tDist, rrz = wz * tDist;
    float tvx = bvx + (boy * rrz - boz * rry);
    float tvy = bvy + (boz * rrx - box * rrz);
    float tvz = bvz + (box * rry - boy * rrx);
    fx += K * (tpx - pi_x) + D * (tvx - vi_x);
    fy += K * (tpy - pi_y) + D * (tvy - vi_y);
    fz += K * (tpz - pi_z) + D * (tvz - vi_z);
  }

  // ---- Brute-force collision O(N²) with shared-memory tiling ----
  // All threads (including inactive) participate in cooperative tile loads
  // so that every thread reaches __syncthreads.  Each particle's data is
  // loaded once per tile and reused by all 256 threads in the block,
  // reducing global memory traffic by ~256×.
  float k_pen  = d_params.stiffness;
  float k_damp = d_params.collisionDamping;
  float k_coh  = d_params.cohesion;
  float r_cut  = d_params.cohesionRadius;
  float mu_p   = d_params.particleFriction;

  for (int tile = 0; tile < shellN; tile += 256) {
    int loadIdx = tile + threadIdx.x;
    if (loadIdx < shellN) {
      sh_px[threadIdx.x] = posX[loadIdx];
      sh_py[threadIdx.x] = posY[loadIdx];
      sh_pz[threadIdx.x] = posZ[loadIdx];
      sh_vx[threadIdx.x] = velX[loadIdx];
      sh_vy[threadIdx.x] = velY[loadIdx];
      sh_vz[threadIdx.x] = velZ[loadIdx];
      sh_r[threadIdx.x]  = radiusArr[loadIdx];
      sh_st[threadIdx.x] = reinterpret_cast<const int*>(state)[loadIdx];
    } else
      sh_st[threadIdx.x] = -1;
         
    __syncthreads();

    if (active) {
      int tileEnd = min(256, shellN - tile);
      for (int k = 0; k < tileEnd; k++) {
        int j = tile + k;
        if (j == tid) continue;
        if (sh_st[k] != ACTIVE) continue;
        float dx_ij = sh_px[k] - pi_x;
        float dy_ij = sh_py[k] - pi_y;
        float dz_ij = sh_pz[k] - pi_z;
        float dist2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
        float rj = sh_r[k];
        float sumR = ri + rj;
        float cutoff = (k_coh > 0.0f && r_cut > sumR) ? r_cut : sumR;
        if (dist2 >= cutoff * cutoff || dist2 < 1e-12f) continue;

        float invDist = rsqrtf(dist2);
        float dist_c = dist2 * invDist;
        float n_x = dx_ij * invDist, n_y = dy_ij * invDist, n_z = dz_ij * invDist;
        float cfx = 0.0f, cfy = 0.0f, cfz = 0.0f;

        if (dist_c < sumR) {
          float pen = sumR - dist_c;
          float vrel_x = sh_vx[k] - vi_x;
          float vrel_y = sh_vy[k] - vi_y;
          float vrel_z = sh_vz[k] - vi_z;
          float vn = vrel_x * n_x + vrel_y * n_y + vrel_z * n_z;
          float f_mag = fmaxf(k_pen * pen - k_damp * vn, 0.0f);
          cfx = f_mag * n_x; cfy = f_mag * n_y; cfz = f_mag * n_z;
          if (mu_p > 0.0f) {
            float vt_x = vrel_x - vn * n_x;
            float vt_y = vrel_y - vn * n_y;
            float vt_z = vrel_z - vn * n_z;
            float vt2 = vt_x * vt_x + vt_y * vt_y + vt_z * vt_z;
            if (vt2 > 1e-16f) {
              float invVt = rsqrtf(vt2);
              float fFric = mu_p * fabsf(f_mag);
              cfx -= fFric * vt_x * invVt;
              cfy -= fFric * vt_y * invVt;
              cfz -= fFric * vt_z * invVt;
            }
          }
        } else if (k_coh > 0.0f) {
          float coh_mag = k_coh * (1.0f - dist_c / r_cut);
          cfx = -coh_mag * n_x; cfy = -coh_mag * n_y; cfz = -coh_mag * n_z;
        }
        fx += cfx; fy += cfy; fz += cfz;
      }
    }
    __syncthreads();
  }

  if (!active) return;

  // ---- Semi-implicit Euler integration ----
  float dt    = d_params.dt;
  float inv_m = 1.0f / m;
  float vx = vi_x + fx * inv_m * dt;
  float vy = vi_y + fy * inv_m * dt;
  float vz = vi_z + fz * inv_m * dt;
  float px = pi_x + vx * dt;
  float py = pi_y + vy * dt;
  float pz = pi_z + vz * dt;

  // ---- Ground collision (inclined plane) ----
  float sn = d_params.slopeSin;
  float cs = d_params.slopeCos;
  float H  = d_params.slopeHeight;
  float gdist = px * sn + py * cs - H;
  if (gdist < ri) {
    float gpen = ri - gdist;
    px += gpen * sn; py += gpen * cs;
    float vn = vx * sn + vy * cs;
    if (vn < 0.0f) {
      float e   = d_params.restitution;
      float dvn = -(1.0f + e) * vn;
      float vtX = vx - vn * sn, vtY = vy - vn * cs, vtZ = vz;
      float vt2 = vtX * vtX + vtY * vtY + vtZ * vtZ;
      float mu = d_params.friction;
      float fric = mu * fabsf(dvn);
      if (vt2 > 1e-16f) {
        float invVtMag = rsqrtf(vt2);
        float vtMag = vt2 * invVtMag;
        if (fric < vtMag) {
          float scale = 1.0f - fric * invVtMag;
          vtX *= scale; vtY *= scale; vtZ *= scale;
        } else {
          vtX = 0.0f; vtY = 0.0f; vtZ = 0.0f;
        }
      }
      vx = vtX + (vn + dvn) * sn;
      vy = vtY + (vn + dvn) * cs;
      vz = vtZ;
    }
  }

  posX[tid] = px;
  posY[tid] = py;
  posZ[tid] = pz;
  
  velX[tid] = vx;
  velY[tid] = vy;
  velZ[tid] = vz;
}

/**
 * @brief Launch the fused shell physics kernel for small-N shells.
 *        Replaces separate launches for tether + collision + ground forces.
 * 
 * @param ps          ParticleSystem with shell particle data (pos/vel/mass/radius/state).
 * @param ball        SnowballState with current ball state (pos/vel/omega/quat).
 * @param ballRadius  Current radius of the snowball.
 * @param shellN      Number of shell particles.
 * @param stream      CUDA stream for async execution.
 * 
 * @note This kernel is optimized for small shell counts (≤ BF_COLLISION_THRESHOLD) where the overhead of building the spatial grid outweighs the benefits.
 *       For larger shell counts, the separate kernels with spatial partitioning should be used instead.
 */
void launchShellPhysicsFused(ParticleSystem &ps, const SnowballState &ball, float ballRadius, int shellN, cudaStream_t stream) {
  if (shellN <= 0) return;

  const int threadsPerBlock = 256;
  shellPhysicsFusedKernel<<<gridSize(shellN, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    ps.posX, ps.posY, ps.posZ,
    ps.velX, ps.velY, ps.velZ,
    ps.mass, ps.radius, ps.state,
    ps.attachLocalX, ps.attachLocalY, ps.attachLocalZ,
    ball.posX, ball.posY, ball.posZ,
    ball.velX, ball.velY, ball.velZ,
    ball.omegaX, ball.omegaY, ball.omegaZ,
    ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
    ballRadius, shellN);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused kernel for gravity + damping + tether forces.
 *        This kernel combines the applyForcesKernel and shellCoreTetherKernel into
 *        a single pass. It computes gravity and linear damping, then applies the
 *        spring-damper tether force to pull particles toward the ball surface.
 *
 * @param forceX       Output force X (overwritten).
 * @param forceY       Output force Y (overwritten).
 * @param forceZ       Output force Z (overwritten).
 * @param posX         Particle position X.
 * @param posY         Particle position Y.
 * @param posZ         Particle position Z.
 * @param velX         Particle velocity X.
 * @param velY         Particle velocity Y.
 * @param velZ         Particle velocity Z.
 * @param mass         Per-particle mass.
 * @param state        Per-particle state (only ACTIVE particles processed).
 * @param attachLocalX Body-frame attachment direction X.
 * @param attachLocalY Body-frame attachment direction Y.
 * @param attachLocalZ Body-frame attachment direction Z.
 * @param radiusArr    Per-particle radius.
 * @param bpx          Ball centre X.
 * @param bpy          Ball centre Y.
 * @param bpz          Ball centre Z.
 * @param bvx          Ball velocity X.
 * @param bvy          Ball velocity Y.
 * @param bvz          Ball velocity Z.
 * @param box          Ball angular velocity X.
 * @param boy          Ball angular velocity Y.
 * @param boz          Ball angular velocity Z.
 * @param bqw          Ball quaternion W.
 * @param bqx          Ball quaternion X.
 * @param bqy          Ball quaternion Y.
 * @param bqz          Ball quaternion Z.
 * @param ballRadius   Current ball radius.
 * @param shellN       Number of shell particles.
 * 
 * @note This kernel overwrites the force arrays (does not accumulate)
 *       since it is intended to be launched as the first force stage.
 *       If additional forces are needed, they should be added to this kernel
 *       or a separate accumulation step should be performed after.
 */
__global__ __launch_bounds__(256, 8)
void forceTetherKernel(
    float *__restrict__ forceX, float *__restrict__ forceY, float *__restrict__ forceZ,
    const float *__restrict__ posX,  const float *__restrict__ posY,  const float *__restrict__ posZ,
    const float *__restrict__ velX,  const float *__restrict__ velY,  const float *__restrict__ velZ,
    const float *__restrict__ mass, 
    const ShellParticleState *__restrict__ state,
    const float *__restrict__ attachLocalX, const float *__restrict__ attachLocalY, const float *__restrict__ attachLocalZ,
    const float *__restrict__ radiusArr,
    float bpx, float bpy, float bpz,
    float bvx, float bvy, float bvz,
    float box, float boy, float boz,
    float bqw, float bqx, float bqy, float bqz,
    float ballRadius,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shellN) return;
  if (__ldg(reinterpret_cast<const int*>(&state[tid])) != ACTIVE) return;

  float m  = __ldg(&mass[tid]);
  float kd = d_params.damping;

  // Read particle state once
  float vi_x = __ldg(&velX[tid]);
  float vi_y = __ldg(&velY[tid]);
  float vi_z = __ldg(&velZ[tid]);
  float pi_x = __ldg(&posX[tid]);
  float pi_y = __ldg(&posY[tid]);
  float pi_z = __ldg(&posZ[tid]);

  // --- Gravity + linear damping ---
  // Damping model: F_damp = -kd * m * v
  float fx = -kd * m * vi_x;
  float fy = -m * d_params.gravity - kd * m * vi_y;
  float fz = -kd * m * vi_z;

  // --- Tether spring-damper ---
  float K = d_params.shellTetherK;
  float D = d_params.shellTetherDamp;

  float lx = __ldg(&attachLocalX[tid]);
  float ly = __ldg(&attachLocalY[tid]);
  float lz = __ldg(&attachLocalZ[tid]);

  float wx, wy, wz;
  quatRotateVec(bqw, bqx, bqy, bqz, lx, ly, lz, wx, wy, wz);

  float pr    = __ldg(&radiusArr[tid]);
  float tDist = ballRadius + pr;

  float tpx = bpx + wx * tDist;
  float tpy = bpy + wy * tDist;
  float tpz = bpz + wz * tDist;

  float rx = wx * tDist, ry = wy * tDist, rz = wz * tDist;
  float tvx = bvx + (boy * rz - boz * ry);
  float tvy = bvy + (boz * rx - box * rz);
  float tvz = bvz + (box * ry - boy * rx);

  fx += K * (tpx - pi_x) + D * (tvx - vi_x);
  fy += K * (tpy - pi_y) + D * (tvy - vi_y);
  fz += K * (tpz - pi_z) + D * (tvz - vi_z);

  forceX[tid] = fx;
  forceY[tid] = fy;
  forceZ[tid] = fz;
}

/**
 * @brief Launch the tether force kernel for shell particles.
 * 
 * @param shell   ParticleSystem representing the shell (reads pos/vel/radius, writes force*).
 * @param ball    SnowballState with current ball state for tether target.
 * @param shellN  Number of shell particles.
 * @param stream  CUDA stream for async execution.
 */
void launchForcesTether(ParticleSystem &shell, const SnowballState &ball, int shellN, cudaStream_t stream) {
  if (shellN <= 0) return;

  const int threadsPerBlock = 256;
  forceTetherKernel<<<gridSize(shellN, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    shell.forceX, shell.forceY, shell.forceZ,
    shell.posX, shell.posY, shell.posZ,
    shell.velX, shell.velY, shell.velZ,
    shell.mass,
    shell.state,
    shell.attachLocalX, shell.attachLocalY, shell.attachLocalZ,
    shell.radius,
    ball.posX, ball.posY, ball.posZ,
    ball.velX, ball.velY, ball.velZ,
    ball.omegaX, ball.omegaY, ball.omegaZ,
    ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
    ball.radius,
    shellN);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused integration + ground collision kernel.
 *        This kernel performs semi-implicit Euler integration and then applies the
 *        inclined plane collision response in a single pass. This avoids the need
 *        to write updated positions and velocities to global memory and read them
 *        back for the ground collision step, which can improve performance by
 *        reducing memory bandwidth usage and eliminating one kernel launch.
 *
 * @param posX    Particle position X (read/write).
 * @param posY    Particle position Y (read/write).
 * @param posZ    Particle position Z (read/write).
 * @param velX    Particle velocity X (read/write).
 * @param velY    Particle velocity Y (read/write).
 * @param velZ    Particle velocity Z (read/write).
 * @param forceX  Net force X.
 * @param forceY  Net force Y.
 * @param forceZ  Net force Z.
 * @param mass    Per-particle mass.
 * @param radius  Per-particle radius.
 * @param state   Per-particle state (only ACTIVE particles processed).
 * @param shellN  Number of shell particles.
 * 
 * @note This kernel assumes that the force arrays have already been computed
 *       (e.g., by a previous force kernel) and will overwrite the pos/vel arrays
 *       with the integrated and collision-resolved values.
 */
__global__ __launch_bounds__(256, 8)
void integrateGroundKernel(
    float *__restrict__ posX,  float *__restrict__ posY,  float *__restrict__ posZ,
    float *__restrict__ velX,  float *__restrict__ velY,  float *__restrict__ velZ,
    const float *__restrict__ forceX, const float *__restrict__ forceY, const float *__restrict__ forceZ,
    const float *__restrict__ mass,
    const float *__restrict__ radius,
    const ShellParticleState *__restrict__ state,
    int shellN)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= shellN) return;
  if (__ldg(reinterpret_cast<const int*>(&state[tid])) != ACTIVE) return;

  float dt    = d_params.dt;
  float inv_m = 1.0f / __ldg(&mass[tid]);
  float r     = __ldg(&radius[tid]);

  // Semi-implicit Euler: update velocity then position
  float vx = velX[tid] + __ldg(&forceX[tid]) * inv_m * dt;
  float vy = velY[tid] + __ldg(&forceY[tid]) * inv_m * dt;
  float vz = velZ[tid] + __ldg(&forceZ[tid]) * inv_m * dt;

  float px = posX[tid] + vx * dt;
  float py = posY[tid] + vy * dt;
  float pz = posZ[tid] + vz * dt;

  // --- Ground collision (inclined plane) ---
  float sn = d_params.slopeSin;
  float cs = d_params.slopeCos;
  float H  = d_params.slopeHeight;

  float dist = px * sn + py * cs - H;

  if (dist < r) {
    float pen = r - dist;
    px += pen * sn;
    py += pen * cs;

    float vn = vx * sn + vy * cs;
    if (vn < 0.0f) {
      float e   = d_params.restitution;
      float dvn = -(1.0f + e) * vn;

      float vtX = vx - vn * sn;
      float vtY = vy - vn * cs;
      float vtZ = vz;
      float vt2 = vtX * vtX + vtY * vtY + vtZ * vtZ;

      float mu   = d_params.friction;
      float fric = mu * fabsf(dvn);

      if (vt2 > 1e-16f) {
        float invVtMag = rsqrtf(vt2);
        float vtMag = vt2 * invVtMag;
        if (fric < vtMag) {
          float scale = 1.0f - fric * invVtMag;
          vtX *= scale;
          vtY *= scale;
          vtZ *= scale;
        } else {
          vtX = 0.0f;
          vtY = 0.0f;
          vtZ = 0.0f;
        }
      }

      vx = vtX + (vn + dvn) * sn;
      vy = vtY + (vn + dvn) * cs;
      vz = vtZ;
    }
  }

  posX[tid] = px;
  posY[tid] = py;
  posZ[tid] = pz;
  
  velX[tid] = vx;
  velY[tid] = vy;
  velZ[tid] = vz;
}

/**
 * @brief Launch the fused integration + ground collision kernel.
 * 
 * @param ps      ParticleSystem (reads force/mass/radius/state, writes pos/vel).
 * @param shellN  Number of shell particles.
 * @param stream  CUDA stream for async execution.
 */
void launchIntegrateGround(ParticleSystem &ps, int shellN, cudaStream_t stream) {
  if (shellN <= 0) return;

  const int threadsPerBlock = 256;
  integrateGroundKernel<<<gridSize(shellN, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    ps.posX, ps.posY, ps.posZ,
    ps.velX, ps.velY, ps.velZ,
    ps.forceX, ps.forceY, ps.forceZ,
    ps.mass,
    ps.radius,
    ps.state,
    shellN);
  CUDA_CHECK(cudaGetLastError());
}