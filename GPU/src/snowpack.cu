// NOTE: d_params (__constant__) is defined in helpers.cuh and shared across all .cu files via Unity Build (CMake >= 3.16) or gpu_unity.cu fallback.

#include "include/helpers.cuh"
#include "include/kernel.cuh"
#include "include/snowpack.cuh"

#include <cub/cub.cuh>

#include <curand_kernel.h>

// On CUDA 10.x (Jetson Nano), __ldg is not pulled into the include chain by cuda_runtime.h.
// Placed here - AFTER all system headers
#if (__CUDACC_VER_MAJOR__ < 11) && !defined(__ldg)
  #define __ldg(p) (*(p))
#endif

// anonymous namespace for helper functions and constants
namespace SnowpackNS {
    
  // Force a safer cap on Jetson Nano (SM 5.3) where these kernels can exceed per-block resource limits
  int blockSizeForCurrentGpu() {
    int blockSize = 1024;

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        // Jetson Nano (SM 5.3) has very limited registers/shared memory → smaller blocks
        int JNblockSize = 256;
        if (prop.major == 5 && prop.minor == 3 && blockSize > JNblockSize) blockSize = JNblockSize;
        if (blockSize > prop.maxThreadsPerBlock) blockSize = prop.maxThreadsPerBlock;
      }
    }

    return blockSize;
  }
}

// ============================================================================
// Memory management
// ============================================================================
/**
 * @brief Allocate device memory for all Snowpack SoA arrays.
 * 
 * @param sp     Snowpack struct whose device pointers will be allocated.
 * @param N      Number of snowpack particles to allocate.
 */
void allocateSnowpack(Snowpack& sp, int N) {
  sp.count = N;
  size_t fb = N * sizeof(float);
  size_t ib = N * sizeof(int);

  CUDA_CHECK(cudaMalloc(&sp.posX,    fb));
  CUDA_CHECK(cudaMalloc(&sp.posY,    fb));
  CUDA_CHECK(cudaMalloc(&sp.posZ,    fb));
  CUDA_CHECK(cudaMalloc(&sp.radius,  fb));
  CUDA_CHECK(cudaMalloc(&sp.mass,    fb));
  CUDA_CHECK(cudaMalloc(&sp.wetness, fb));
  CUDA_CHECK(cudaMalloc(&sp.alive,   ib));
}

/**
 * @brief Free device memory for a Snowpack.
 * 
 * @param sp  Snowpack to deallocate (count set to 0).
 */
void freeSnowpack(Snowpack& sp) {
  cudaFree(sp.posX);
  cudaFree(sp.posY);
  cudaFree(sp.posZ);
  cudaFree(sp.radius);
  cudaFree(sp.mass);
  cudaFree(sp.wetness);
  cudaFree(sp.alive);
  
  sp.count = 0;
}

// ============================================================================
// sortSnowpackByPosX - one-time sort at init for range-limited capture
//
// Sorts ALL snowpack SoA arrays by ascending posX using CUB radix sort.
// After sorting, the capture kernel only needs to process the narrow
// X-range around the ball, reducing wasted threads by ~50-100×.
//
// Uses fillIotaKernel, gatherFloatKernel, and gatherIntKernel to implement the sort + gather pattern
// NOTE: This is a one-time cost at initialization, amortized over many frames of capture.
// ============================================================================

/**
 * @brief Fill an array with iota (0, 1, 2, ...) for given length N.
 * 
 * @param arr  Array to fill.
 * @param N    Number of elements.
 */
__global__ void fillIotaKernel(uint32_t *arr, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) arr[tid] = (uint32_t)tid;
}

/**
 * @brief Gather elements from src to dst using indices (float version).
 * 
 * @param dst       Output array to write gathered elements.
 * @param src       Input array to read from.
 * @param indices   Array of indices to gather.
 * @param N         Number of elements to gather.
 */
__global__ void gatherFloatKernel(
    float *__restrict__ dst, const float *__restrict__ src,
    const uint32_t *__restrict__ indices,
    int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) dst[tid] = src[indices[tid]];
}

/**
 * @brief Gather elements from src to dst using indices (int version).
 * 
 * @param dst       Output array to write gathered elements.
 * @param src       Input array to read from.
 * @param indices   Array of indices to gather.
 * @param N         Number of elements to gather.
 */
__global__ void gatherIntKernel(
    int *__restrict__ dst, const int *__restrict__ src,
    const uint32_t *__restrict__ indices,
    int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) dst[tid] = src[indices[tid]];
}

/**
 * @brief Sort snowpack particles by ascending posX using CUB radix sort.
 *        Sorts ALL SoA arrays to maintain alignment.
 * 
 * @param sp  Snowpack to sort (device pointers and count must be set).
 */
void sortSnowpackByPosX(Snowpack& sp) {
  if (sp.count <= 1) return;
  
  // --- Allocate sort buffers ---
  float    *d_keysAlt;
  uint32_t *d_valsIn, *d_valsAlt;
  CUDA_CHECK(cudaMalloc(&d_keysAlt,  sp.count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_valsIn,   sp.count * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_valsAlt,  sp.count * sizeof(uint32_t)));

  const int threadsPerBlock1 = 256;
  fillIotaKernel<<<gridSize(sp.count, threadsPerBlock1), threadsPerBlock1>>>(d_valsIn, sp.count);
  CUDA_CHECK(cudaGetLastError());

  // --- CUB radix sort (posX, index) pairs ---
  cub::DoubleBuffer<float>    d_keys(sp.posX, d_keysAlt);
  cub::DoubleBuffer<uint32_t> d_vals(d_valsIn, d_valsAlt);
  size_t tempBytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, d_keys, d_vals, sp.count);
  void *d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
  cub::DeviceRadixSort::SortPairs(d_temp, tempBytes, d_keys, d_vals, sp.count);
  CUDA_CHECK(cudaDeviceSynchronize());

  float    *sortedPosX    = d_keys.Current();
  uint32_t *sortedIndices = d_vals.Current();

  // If CUB placed sorted posX in the alt buffer, copy to sp.posX
  if (sortedPosX != sp.posX) CUDA_CHECK(cudaMemcpy(sp.posX, sortedPosX, sp.count * sizeof(float), cudaMemcpyDeviceToDevice));

  // --- Gather remaining arrays (one at a time to minimise temp memory) ---
  float *d_tmpF;
  CUDA_CHECK(cudaMalloc(&d_tmpF, sp.count * sizeof(float)));

  auto gatherF = [&](float *arr) {
    CUDA_CHECK(cudaMemcpy(d_tmpF, arr, sp.count * sizeof(float), cudaMemcpyDeviceToDevice));
    const int threadsPerBlock2 = 256;
    gatherFloatKernel<<<gridSize(sp.count, threadsPerBlock2), threadsPerBlock2>>>(arr, d_tmpF, sortedIndices, sp.count);
    CUDA_CHECK(cudaGetLastError());
  };

  gatherF(sp.posY); gatherF(sp.posZ);
  gatherF(sp.radius);
  gatherF(sp.mass);
  gatherF(sp.wetness);

  // Gather alive (int-sized enum)
  CUDA_CHECK(cudaMemcpy(d_tmpF, sp.alive, sp.count * sizeof(int), cudaMemcpyDeviceToDevice));
  const int threadsPerBlock3 = 256;
  gatherIntKernel<<<gridSize(sp.count, threadsPerBlock3), threadsPerBlock3>>>(
      reinterpret_cast<int*>(sp.alive),
      reinterpret_cast<const int*>(d_tmpF),
      sortedIndices, sp.count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Cleanup ---
  cudaFree(d_keysAlt);
  cudaFree(d_valsIn);
  cudaFree(d_valsAlt);
  cudaFree(d_temp);
  cudaFree(d_tmpF);

  printf("[Snowpack] Sorted %d particles by posX for range-limited capture\n", sp.count);
}

// ============================================================================
// initSnowpackKernel - place particles in a 3D rectangular band on the slope
//
// Slope geometry (same convention as simulation.cu):
//   Normal N = (sinθ, cosθ, 0)              - points UP from surface
//   Downhill tangent T = (cosθ, -sinθ, 0)   - X increases, Y decreases
//   Plane: sinθ·x + cosθ·y = slopeHeight
//   Origin on slope: (0, H/cosθ)
//
// Band parameterised by (s, z, n):
//   s ∈ [spawnStartS, spawnStartS + spawnLengthS]   - along slope
//   z ∈ [-spawnWidthZ/2, +spawnWidthZ/2]            - lateral
//   n ∈ [0, spawnThicknessN]                        - normal (stacking layers)
// ============================================================================
/**
 * @brief Place snowpack particles in a 3D rectangular band on the slope.
 *        Band parameterised by (s, z, n): along-slope, lateral, and normal layers.
 *        Uses the same slope convention as simulation.cu.
 *
 * @param posX    Output X positions.
 * @param posY    Output Y positions.
 * @param posZ    Output Z positions.
 * @param radius  Output per-particle radius.
 * @param mass    Output per-particle mass.
 * @param wetness Output per-particle wetness (randomised in [wetnessMin, wetnessMax]).
 * @param alive   Output alive flag.
 * @param seed    Random seed for jitter and wetness variation (can be frame-based).
 */
__global__ void initSnowpackKernel(
    float *__restrict__ posX, float *__restrict__ posY, float *__restrict__ posZ,
    float *__restrict__ radius,
    float *__restrict__ mass,
    float *__restrict__ wetness,
    SnowpackParticleState *__restrict__ alive,
    unsigned int seed)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d_params.numParticles) return;

  curandState rng;
  curand_init(seed + tid, 0, 0, &rng);  // tid in seed (not sequence) → O(1) init

  float sn = d_params.slopeSin;
  float cs = d_params.slopeCos;
  float H  = d_params.slopeHeight;
  float Ls = d_params.spawnLengthS;
  float Wz = d_params.spawnWidthZ;
  float S0 = d_params.spawnStartS;
  float Tn = d_params.spawnThicknessN;
  float pr = d_params.particleRadius;

  // Compute layers along normal
  int layersN = 1;
  if (Tn > 0.0f) layersN = max(1, (int)(Tn / (2.0f * pr)));
  int perLayer = max(1, d_params.numParticles / layersN);
  int layer = tid / perLayer;
  int inLayer = tid % perLayer;
  if (layer >= layersN) {
    layer = layersN - 1; 
    inLayer = perLayer - 1;
  }

  // Grid within one layer
  int cols = max(1, (int)sqrtf((float)perLayer * Wz / Ls));
  int rows = (perLayer + cols - 1) / cols;
  int row = inLayer / cols;
  int col = inLayer % cols;
  if (row >= rows) row = rows - 1;

  float spacingS = Ls / (float)rows;
  float spacingZ = Wz / (float)cols;

  // Position in slope coordinates with small jitter to break up regularity (jitter is up to 30% of spacing)
  float s = S0 + row * spacingS + curand_uniform(&rng) * spacingS * 0.3f;
  float z = -Wz * 0.5f + col * spacingZ + curand_uniform(&rng) * spacingZ * 0.3f;
  float n = pr + layer * 2.0f * pr; // offset along normal (stacked layers)

  // Convert to world coordinates
  float x0 = 0.0f;
  float y0 = H / cs;
  float worldX = x0 + s * cs + n * sn;
  float worldY = y0 - s * sn + n * cs;
  float worldZ = z;
  posX[tid] = worldX;
  posY[tid] = worldY;
  posZ[tid] = worldZ;

  radius[tid] = pr;
  mass[tid]   = d_params.particleMass;

  float wMin = d_params.wetnessMin;
  float wMax = d_params.wetnessMax;
  wetness[tid] = wMin + curand_uniform(&rng) * (wMax - wMin);

  alive[tid] = AVAILABLE;
}

/**
 * @brief Launch the snowpack initialization kernel.
 * 
 * @param sp    Snowpack struct with allocated device arrays.
 * @param seed  Random seed for jitter and wetness variation.
 */
void launchInitSnowpack(Snowpack& sp, unsigned int seed) {
  const int threadsPerBlock = SnowpackNS::blockSizeForCurrentGpu();
  initSnowpackKernel<<<gridSize(sp.count, threadsPerBlock), threadsPerBlock>>>(
    sp.posX, sp.posY, sp.posZ,
    sp.radius,
    sp.mass,
    sp.wetness,
    sp.alive,
    seed);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// captureFromSnowpackKernel - O(N_snowpack) scan per frame
// ============================================================================
/**
 * @brief Capture snowpack particles near the ball and append them to the shell.
 *        Each thread processes one snowpack particle.  Steps:
 *            1. Skip if already consumed (alive != AVAILABLE).
 *            2. Distance check against ball.
 *            3. Probabilistic sticking (sigmoid with logit clamp).
 *            4. Atomic-reserve a slot in the shell buffer.
 *            5. Write captured data into shell.
 *            6. Mark snowpack particle consumed.
 *            7. Accumulate captured mass.
 *
 * @param sp_posX           Snowpack position X.
 * @param sp_posY           Snowpack position Y.
 * @param sp_posZ           Snowpack position Z.
 * @param sp_radius         Snowpack per-particle radius.
 * @param sp_mass           Snowpack per-particle mass.
 * @param sp_wetness        Snowpack per-particle wetness.
 * @param sp_alive          Snowpack alive flag.
 * @param N                 Total number of snowpack particles.
 * @param sh_posX           Shell output position X.
 * @param sh_posY           Shell output position Y.
 * @param sh_posZ           Shell output position Z.
 * @param sh_velX           Shell output velocity X.
 * @param sh_velY           Shell output velocity Y.
 * @param sh_velZ           Shell output velocity Z.
 * @param sh_forceX         Shell output force X (set to 0).
 * @param sh_forceY         Shell output force Y (set to 0).
 * @param sh_forceZ         Shell output force Z (set to 0).
 * @param sh_mass           Shell output mass.
 * @param sh_radius         Shell output radius.
 * @param sh_state          Shell output state (set to ACTIVE).
 * @param sh_wetness        Shell output wetness.
 * @param sh_attachLocalX   Shell output body-frame direction X.
 * @param sh_attachLocalY   Shell output body-frame direction Y.
 * @param sh_attachLocalZ   Shell output body-frame direction Z.
 * @param shellN            Current number of shell particles.
 * @param bpx               Ball centre X.
 * @param bpy               Ball centre Y.
 * @param bpz               Ball centre Z.
 * @param bvx               Ball velocity X.
 * @param bvy               Ball velocity Y.
 * @param bvz               Ball velocity Z.
 * @param box               Ball angular velocity X.
 * @param boy               Ball angular velocity Y.
 * @param boz               Ball angular velocity Z.
 * @param bqw               Ball quaternion W.
 * @param bqx               Ball quaternion X.
 * @param bqy               Ball quaternion Y.
 * @param bqz               Ball quaternion Z.
 * @param ballRadius        Current ball radius.
 * @param d_captureCount    Device counter for captures this frame.
 * @param d_captureMass     Device accumulator for captured mass.
 * @param frame             Current simulation frame (varies hash).
 * @param shellCapacity     Maximum shell capacity (for bounds checking).
 * @param spOffset         Offset into snowpack arrays to start processing (for splitting into multiple launches).
 * @param spCount          Number of snowpack particles to process in this launch (for splitting into multiple launches).
 */
__global__ __launch_bounds__(256, 5)
void captureFromSnowpackKernel(
    const float *__restrict__ sp_posX, const float *__restrict__ sp_posY, const float *__restrict__ sp_posZ,
    const float *__restrict__ sp_radius,
    const float *__restrict__ sp_mass,
    const float *__restrict__ sp_wetness,
    SnowpackParticleState *__restrict__ sp_alive,
    int N,
    float *__restrict__ sh_posX,  float *__restrict__ sh_posY,  float *__restrict__ sh_posZ,
    float *__restrict__ sh_velX,  float *__restrict__ sh_velY,  float *__restrict__ sh_velZ,
    float *__restrict__ sh_forceX,float *__restrict__ sh_forceY,float *__restrict__ sh_forceZ,
    float *__restrict__ sh_mass, 
    float *__restrict__ sh_radius,
    ShellParticleState *__restrict__ sh_state,
    float *__restrict__ sh_wetness,
    float *__restrict__ sh_attachLocalX, float *__restrict__ sh_attachLocalY, float *__restrict__ sh_attachLocalZ,
    int shellN,
    float bpx, float bpy, float bpz,
    float bvx, float bvy, float bvz,
    float box, float boy, float boz,
    float bqw, float bqx, float bqy, float bqz,
    float ballRadius,
    int *__restrict__ d_captureCount,
    float *__restrict__ d_captureMass,
    int frame,
    int shellCapacity,
    int spOffset,
    int spCount)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= spCount) return;
  if (sp_alive[tid + spOffset] != AVAILABLE) return;

  float px = sp_posX[tid + spOffset], py = sp_posY[tid + spOffset], pz = sp_posZ[tid + spOffset];
  
  float pr = sp_radius[tid + spOffset];

  // Distance check
  float dx = px - bpx;
  float dy = py - bpy;
  float dz = pz - bpz;
  float dist2 = dx * dx + dy * dy + dz * dz;

  float captureR = ballRadius + pr + 0.3f;  // ball surface + particle radius + margin
  float captureR2 = captureR * captureR;
  if (dist2 > captureR2 || dist2 < 1e-16f) return;

  // Relative velocity (snowpack is static → vrel = -ball velocity)
  float vrelMag = sqrtf(bvx * bvx + bvy * bvy + bvz * bvz);

  // Probabilistic sticking (sigmoid with logit clamp)
  float w = sp_wetness[tid + spOffset];
  float logit = d_params.stickK0
              + d_params.stickK1 * w
              - d_params.stickK2 * vrelMag
              + d_params.stickRadiusBoost * ballRadius;

  // Clamp logit for numerical stability and tuning
  float clamp = d_params.logitClamp;
  logit = fminf(fmaxf(logit, -clamp), clamp);

  float P_stick = __fdividef(1.0f, (1.0f + __expf(-logit)));

  // Per-thread hash-based PRNG (varies per frame)
  uint32_t h = (uint32_t)(tid + spOffset) ^ (uint32_t)(frame * 2654435761u);
  h ^= h >> 16; h *= 0x45d9f3bu;
  h ^= h >> 16; h *= 0x45d9f3bu;
  h ^= h >> 16;
  float rnd = (float)(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
  if (rnd >= P_stick) return;  // didn't stick

  // Reserve a slot in the shell
  int slot = atomicAdd(d_captureCount, 1);
  if ((d_params.maxCapturePerFrm > 0 && slot >= d_params.maxCapturePerFrm) ||
      (shellN + slot >= shellCapacity)) return; // per-frame cap, capacity overflow

  // Mark consumed
  sp_alive[tid + spOffset] = CONSUMED;

  // Accumulate captured mass AFTER all validity checks so that d_captureMass
  // reflects only particles that are actually written into the shell.
  float mass = sp_mass[tid + spOffset];
  atomicAdd(d_captureMass, mass);

  // Compute shell particle data
  int si = shellN + slot;  // shell index

  // Direction from ball center to particle
  float invDist = rsqrtf(dist2);
  float nx = dx * invDist, ny = dy * invDist, nz = dz * invDist;

  float surfDist = ballRadius + pr;
  float rx = nx * surfDist, ry = ny * surfDist, rz = nz * surfDist;

  // Place particle on ball surface
  sh_posX[si] = bpx + rx;
  sh_posY[si] = bpy + ry;
  sh_posZ[si] = bpz + rz;

  // Set velocity to ball surface velocity: v_ball + ω × r
  sh_velX[si] = bvx + (boy * rz - boz * ry); 
  sh_velY[si] = bvy + (boz * rx - box * rz);
  sh_velZ[si] = bvz + (box * ry - boy * rx);

  sh_forceX[si] = 0.0f;
  sh_forceY[si] = 0.0f;
  sh_forceZ[si] = 0.0f;

  sh_mass[si]    = mass;
  sh_radius[si]  = pr;
  sh_state[si]   = ACTIVE;
  sh_wetness[si] = w;

  // Store UNIT direction in body frame as tether anchor
  // (scales with ball.radius at runtime so shell expands as ball grows)
  quatInvRotateVec(bqw, bqx, bqy, bqz,
                   nx, ny, nz,
                   sh_attachLocalX[si], sh_attachLocalY[si], sh_attachLocalZ[si]);
}

/**
 * @brief Launch the capture kernel: scan snowpack, capture particles near ball, append to shell.
 * 
 * @param sp                Snowpack (reads pos/mass/wetness, marks alive=CONSUMED on capture).
 * @param shell             Shell ParticleSystem to append captured particles to.
 * @param shellN            Current number of active shell particles.
 * @param ball              Current snowball state.
 * @param d_captureCount    Device pointer: atomically incremented per capture.
 * @param d_captureMass     Device pointer: atomically accumulated captured mass.
 * @param frame             Current frame number (used for PRNG variation).
 * @param shellCapacity     Maximum shell capacity (for bounds checking).
 * @param spOffset          Offset into snowpack arrays to start processing (for splitting into multiple launches).
 * @param spCount           Number of snowpack particles to process in this launch (for splitting into multiple launches).
 * @param stream            CUDA stream for async execution.
 */
void launchCaptureFromSnowpack(
    Snowpack& sp,
    ParticleSystem& shell, int shellN,
    const SnowballState& ball,
    int* d_captureCount,
    float* d_captureMass,
    int frame,
    int shellCapacity,
    int spOffset,
    int spCount,
    cudaStream_t stream)
{
  int effectiveCount  = (spCount >= 0) ? spCount : sp.count;
  int effectiveOffset = (spCount >= 0) ? spOffset : 0;
  if (effectiveCount <= 0) return;

  const int threadsPerBlock = 256;  // better for range-limited launches on small SMs
  captureFromSnowpackKernel<<<gridSize(effectiveCount, threadsPerBlock), threadsPerBlock, 0, stream>>>(
    sp.posX, sp.posY, sp.posZ,
    sp.radius,
    sp.mass,
    sp.wetness,
    sp.alive,
    sp.count,
    shell.posX, shell.posY, shell.posZ,
    shell.velX, shell.velY, shell.velZ,
    shell.forceX, shell.forceY, shell.forceZ,
    shell.mass,
    shell.radius,
    shell.state,
    shell.wetness,
    shell.attachLocalX, shell.attachLocalY, shell.attachLocalZ,
    shellN,
    ball.posX, ball.posY, ball.posZ,
    ball.velX, ball.velY, ball.velZ,
    ball.omegaX, ball.omegaY, ball.omegaZ,
    ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
    ball.radius,
    d_captureCount,
    d_captureMass,
    frame,
    shellCapacity,
    effectiveOffset, effectiveCount);
  CUDA_CHECK(cudaGetLastError());
}