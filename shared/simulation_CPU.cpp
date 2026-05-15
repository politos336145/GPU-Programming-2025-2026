#include "include/global.h"
#include "include/logging.h"
#include "include/memory_CPU.h"
#include "include/profiling.h"
#include "include/simulation_CPU.h"
#include "include/types.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

#ifdef HAS_OPENMP
  #include <omp.h>
#endif

// Defined in main - shared timestamp for log filenames.
extern std::string timestamp;

/**
 * @brief Print statistics about the spatial grid occupancy for debugging and tuning.
 * @param grid    GridData to analyze.
 * @param shellN  Number of shell particles (for expected count).
 * @param frame   Current frame number (for logging).
 */
void inline printGridStats(const GridData& grid, int shellN, int frame) {
  int numCells = grid.hashTableSize;

  int nonEmpty = 0;
  int maxBucket = 0;
  long long totalCount = 0;

  for (int c = 0; c < numCells; c++) {
    if (grid.cellStart[c] == INACTIVE) continue; // empty cell

    int count = grid.cellEnd[c] - grid.cellStart[c];
    totalCount += count;
    nonEmpty++;
    if (count > maxBucket) maxBucket = count;
  }

  float avgBucket = (nonEmpty > 0) ? (float)totalCount / nonEmpty : 0.0f;

  printf("[Grid] Frame %d: non-empty = %d / %d, max bucket = %d, avg = %.1f, sum = %lld (expected %d)\n",
          frame, nonEmpty, numCells, maxBucket, avgBucket, totalCount, shellN);
}

// ============================================================================
// Memory management
// ============================================================================

/**
 * @brief Allocate memory for all SoA arrays of a ParticleSystem.
 * @param ps        ParticleSystem whose pointers will be allocated.
 * @param capacity  Maximum number of particles.
 */
void allocateParticleSystem(ParticleSystem &ps, int capacity) {
  ps.capacity = capacity;

  ps.posX         = (float *)ALLOC(capacity * sizeof(float));
  ps.posY         = (float *)ALLOC(capacity * sizeof(float));
  ps.posZ         = (float *)ALLOC(capacity * sizeof(float));
  ps.velX         = (float *)ALLOC(capacity * sizeof(float));
  ps.velY         = (float *)ALLOC(capacity * sizeof(float));
  ps.velZ         = (float *)ALLOC(capacity * sizeof(float));
  ps.forceX       = (float *)ALLOC(capacity * sizeof(float));
  ps.forceY       = (float *)ALLOC(capacity * sizeof(float));
  ps.forceZ       = (float *)ALLOC(capacity * sizeof(float));
  ps.radius       = (float *)ALLOC(capacity * sizeof(float));
  ps.mass         = (float *)ALLOC(capacity * sizeof(float));
  ps.wetness      = (float *)ALLOC(capacity * sizeof(float));
  ps.attachLocalX = (float *)ALLOC(capacity * sizeof(float));
  ps.attachLocalY = (float *)ALLOC(capacity * sizeof(float));
  ps.attachLocalZ = (float *)ALLOC(capacity * sizeof(float));

  ps.state  = (ShellParticleState *)ALLOC(capacity * sizeof(ShellParticleState));
  std::fill_n(ps.state, capacity, INACTIVE);  
}

/**
 * @brief Free all memory owned by a ParticleSystem.
 * @param ps  ParticleSystem to deallocate (capacity set to 0).
 */
void freeParticleSystem(ParticleSystem &ps) {
  FREE(ps.posX);
  FREE(ps.posY);
  FREE(ps.posZ);
  FREE(ps.velX);
  FREE(ps.velY);
  FREE(ps.velZ);
  FREE(ps.forceX);
  FREE(ps.forceY);
  FREE(ps.forceZ);
  FREE(ps.mass);
  FREE(ps.radius);
  FREE(ps.state);
  FREE(ps.wetness);
  FREE(ps.attachLocalX);
  FREE(ps.attachLocalY);
  FREE(ps.attachLocalZ);
  
  ps.capacity = 0;
}

/**
 * @brief Allocate memory for a Snowpack (SoA arrays).
 * @param sp     Snowpack struct whose pointers will be allocated.
 * @param count  Number of snowpack particles.
 */
void allocateSnowpack(Snowpack &sp, int count) {
  sp.count = count;
  size_t fb = count * sizeof(float);
  
  sp.posX    = (float *)ALLOC(fb);
  sp.posY    = (float *)ALLOC(fb);
  sp.posZ    = (float *)ALLOC(fb);
  sp.radius  = (float *)ALLOC(fb);
  sp.mass    = (float *)ALLOC(fb);
  sp.wetness = (float *)ALLOC(fb);
  sp.alive   = (SnowpackParticleState *)ALLOC(count * sizeof(SnowpackParticleState));
}

/**
 * @brief Free all memory owned by a Snowpack.
 * @param sp  Snowpack to deallocate (count set to 0).
 */
void freeSnowpack(Snowpack &sp) {
  FREE(sp.posX);
  FREE(sp.posY);
  FREE(sp.posZ);
  FREE(sp.radius);
  FREE(sp.mass);
  FREE(sp.wetness);
  FREE(sp.alive);

  sp.count = 0;
}

/**
 * @brief Allocate memory for spatial grid data structures.
 * @param grid          GridData struct to allocate.
 * @param params        SimParams providing domain geometry and cell size.
 * @param shellCapacity Current shell capacity (sizes per-particle arrays).
 */
void allocateGrid(GridData &grid, const SimParams &params, int shellCapacity) {
  computeGridDimensions(grid, params);

  // CPU uses direct linear indexing (no modular hashing) so we must keep
  // numCells small enough to allocate cellStart/cellEnd arrays.
  // Cap grid dimensions to stay under ~128 MB for the two cell arrays.
  grid.gridDimX = std::min(grid.gridDimX, 512);
  grid.gridDimY = std::min(grid.gridDimY, 256);
  grid.gridDimZ = std::min(grid.gridDimZ, 128);
  grid.numCells = grid.gridDimX * grid.gridDimY * grid.gridDimZ;

  // no modular hashing - use full numCells for cell arrays
  grid.hashTableSize = grid.numCells;
  grid.hashTableMask = grid.numCells - 1;  // not necessarily power-of-2, uses direct index not mask

  printf("[Grid] dims = %d x %d x %d  (%d cells, cellSize = %.4f m)\n",
         grid.gridDimX, grid.gridDimY, grid.gridDimZ, grid.numCells, grid.cellSize);
  printf("[Grid] origin = (%.2f, %.2f, %.2f)\n", grid.originX, grid.originY, grid.originZ);
  
  // -----------------------------------------------------------------------
  // Allocate per-particle arrays (size = capacity)
  // -----------------------------------------------------------------------
  int N = shellCapacity;
  grid.cellHash      = (uint32_t *)ALLOC(N * sizeof(uint32_t));
  grid.particleIndex = (uint32_t *)ALLOC(N * sizeof(uint32_t));

  // not used in CPU version, but allocate for API compatibility with GPU version
  grid.cellHashAlt      = nullptr;
  grid.particleIndexAlt = nullptr;
  grid.d_sortTemp       = nullptr;
  grid.sortTempBytes    = 0;

  // -----------------------------------------------------------------------
  // Allocate per-cell arrays
  // -----------------------------------------------------------------------
  grid.cellStart = (int *)ALLOC(grid.numCells * sizeof(int));
  grid.cellEnd   = (int *)ALLOC(grid.numCells * sizeof(int));

  // One-time init: mark all cells empty.  Subsequent frames use sparse
  // clearing (buildGrid only resets cells that were previously occupied).
  std::fill_n(grid.cellStart, grid.numCells, INACTIVE);
  std::fill_n(grid.cellEnd,   grid.numCells, INACTIVE);
}

/**
 * @brief Free all memory owned by a GridData.
 * @param grid  GridData to deallocate.
 */
void freeGrid(GridData &grid) {
  FREE(grid.cellHash);
  FREE(grid.particleIndex);
  FREE(grid.cellStart);
  FREE(grid.cellEnd);

  grid.cellHash      = nullptr;
  grid.particleIndex = nullptr;
  grid.cellStart     = nullptr;
  grid.cellEnd       = nullptr;
  grid.numCells      = 0;
}

/**
 * @brief Grow a ParticleSystem to a larger capacity, preserving active data.
 * @param ps           ParticleSystem to grow (modified in-place).
 * @param activeCount  Number of active particles to preserve.
 * @param newCapacity  New capacity (must be > ps.capacity).
 */
void growParticleSystem(ParticleSystem &ps, int activeCount, int newCapacity) {
  auto grow = [&](float *&ptr) {
    float *tmp = (float *)ALLOC(newCapacity * sizeof(float));
    if (activeCount > 0) std::copy_n(ptr, activeCount, tmp);
    FREE(ptr);
    ptr = tmp;
  };

  grow(ps.posX);
  grow(ps.posY);
  grow(ps.posZ);
  
  grow(ps.velX);
  grow(ps.velY);
  grow(ps.velZ);
  
  grow(ps.forceX);
  grow(ps.forceY);
  grow(ps.forceZ);
  
  grow(ps.mass);
  grow(ps.radius);
  grow(ps.wetness);

  grow(ps.attachLocalX);
  grow(ps.attachLocalY);
  grow(ps.attachLocalZ);
  {
    ShellParticleState *tmp = (ShellParticleState *)ALLOC(newCapacity * sizeof(ShellParticleState));
    if (activeCount > 0) std::copy_n(ps.state, activeCount, tmp);
    std::fill_n(tmp + activeCount, newCapacity - activeCount, INACTIVE);
    FREE(ps.state);
    ps.state = tmp;
  }
  ps.capacity = newCapacity;
}

/**
 * @brief Grow the grid to match a new shell capacity.
 * @param grid        GridData to grow.
 * @param newCapacity New capacity for the per-particle arrays.
 */
void growGrid(GridData &grid, int newCapacity) {
  FREE(grid.cellHash);
  FREE(grid.particleIndex);

  grid.cellHash      = (uint32_t *)ALLOC(newCapacity * sizeof(uint32_t));
  grid.particleIndex = (uint32_t *)ALLOC(newCapacity * sizeof(uint32_t));

  // Zero cellHash so the stale sparse-clear in buildGrid (static prevN) reads
  // valid zeros instead of garbage, preventing OOB writes to cellStart/cellEnd.
  std::fill_n(grid.cellHash, newCapacity, (uint32_t)0);
  // Re-initialise cell tables: stale entries from before the grow must be
  // wiped so the next buildGrid starts from a clean state.
  std::fill_n(grid.cellStart, grid.numCells, INACTIVE);
  std::fill_n(grid.cellEnd,   grid.numCells, INACTIVE);
}

// ============================================================================
// Quaternion helpers
// ============================================================================

/**
 * @brief Rotate a vector by a quaternion (q * v * q^{-1}).
 * @param qw  Quaternion scalar part.
 * @param qx  Quaternion x component.
 * @param qy  Quaternion y component.
 * @param qz  Quaternion z component.
 * @param vx  Input vector x.
 * @param vy  Input vector y.
 * @param vz  Input vector z.
 * @param ox  Output rotated vector x.
 * @param oy  Output rotated vector y.
 * @param oz  Output rotated vector z.
 */
void quatRotateVec(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float &ox, float &oy, float &oz) {
  float tx = 2.0f * (qy * vz - qz * vy);
  float ty = 2.0f * (qz * vx - qx * vz);
  float tz = 2.0f * (qx * vy - qy * vx);
  ox = vx + qw * tx + (qy * tz - qz * ty);
  oy = vy + qw * ty + (qz * tx - qx * tz);
  oz = vz + qw * tz + (qx * ty - qy * tx);
}

/**
 * @brief Rotate a vector by the inverse of a quaternion.
 * @param qw  Quaternion scalar part.
 * @param qx  Quaternion x component.
 * @param qy  Quaternion y component.
 * @param qz  Quaternion z component.
 * @param vx  Input vector x.
 * @param vy  Input vector y.
 * @param vz  Input vector z.
 * @param ox  Output rotated vector x.
 * @param oy  Output rotated vector y.
 * @param oz  Output rotated vector z.
 */
void quatInvRotateVec(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float &ox, float &oy, float &oz) {
  quatRotateVec(qw, -qx, -qy, -qz, vx, vy, vz, ox, oy, oz);
}

// ============================================================================
// lcgRandom - Simple LCG PRNG
// ============================================================================
/**
 * @brief Linear congruential generator PRNG.
 * @param state  RNG state, updated in place.
 * @return Random float in [0, 1).
 */
float lcgRandom(uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  return (float)(state & 0x00FFFFFFu) / (float)0x01000000u;
}

// ============================================================================
// neighborCollision - 27-cell traversal, soft-sphere + cohesion.
// ============================================================================
/**
 * @brief Neighbor collision using spatial grid: soft-sphere + cohesion.
 * @param ps      ParticleSystem (reads pos/vel/radius, writes force*).
 * @param grid    GridData with sorted cell hash tables.
 * @param shellN  Number of shell particles.
 * @param params  SimParams providing stiffness, damping, cohesion, friction.
 */
void neighborCollision(ParticleSystem &ps, const GridData &grid, int shellN, const SimParams &params) {
  float k_pen  = params.stiffness;
  float k_damp = params.collisionDamping;
  float k_coh  = params.cohesion;
  float r_cut  = params.cohesionRadius;
  float mu_p   = params.particleFriction;

  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int i = 0; i < shellN; i++) {
    if (ps.state[i] != ACTIVE) continue;

    float pi_x = ps.posX[i];
    float pi_y = ps.posY[i];
    float pi_z = ps.posZ[i];

    float vi_x = ps.velX[i];
    float vi_y = ps.velY[i];
    float vi_z = ps.velZ[i];

    float ri   = ps.radius[i];

    int cx = (int)floorf((pi_x - grid.originX) / grid.cellSize);
    int cy = (int)floorf((pi_y - grid.originY) / grid.cellSize);
    int cz = (int)floorf((pi_z - grid.originZ) / grid.cellSize);
    cx = (std::max)(0, (std::min)(cx, grid.gridDimX - 1));
    cy = (std::max)(0, (std::min)(cy, grid.gridDimY - 1));
    cz = (std::max)(0, (std::min)(cz, grid.gridDimZ - 1));

    for (int dz = -1; dz <= 1; dz++) {
      int nz = cz + dz;
      if (nz < 0 || nz >= grid.gridDimZ) continue;

      for (int dy = -1; dy <= 1; dy++) {
        int ny = cy + dy;
        if (ny < 0 || ny >= grid.gridDimY) continue;

        for (int dx = -1; dx <= 1; dx++) {
          int nx_cell = cx + dx;
          if (nx_cell < 0 || nx_cell >= grid.gridDimX) continue;

          int hash = nz * grid.gridDimX * grid.gridDimY + ny * grid.gridDimX + nx_cell;
          int start = grid.cellStart[hash];
          if (start == INACTIVE) continue;

          int end = grid.cellEnd[hash];

          for (int s = start; s < end; s++) {
            int j = (int)grid.particleIndex[s];
            if (j <= i) continue;
            if (ps.state[j] != ACTIVE) continue;

            float dx_ij = ps.posX[j] - pi_x;
            float dy_ij = ps.posY[j] - pi_y;
            float dz_ij = ps.posZ[j] - pi_z;
            float dist2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

            float rj   = ps.radius[j];
            float sumR = ri + rj;
            float cutoff = (k_coh > 0.0f && r_cut > sumR) ? r_cut : sumR;

            if (dist2 >= cutoff * cutoff || dist2 < 1e-12f) continue;

            float dist = sqrtf(dist2);
            float invDist = 1.0f / dist;
            float n_x = dx_ij * invDist;
            float n_y = dy_ij * invDist;
            float n_z = dz_ij * invDist;

            float fx = 0.0f;
            float fy = 0.0f;
            float fz = 0.0f;

            if (dist < sumR) {
              float pen = sumR - dist;

              float vrel_x = ps.velX[j] - vi_x;
              float vrel_y = ps.velY[j] - vi_y;
              float vrel_z = ps.velZ[j] - vi_z;
              float vn = vrel_x * n_x + vrel_y * n_y + vrel_z * n_z;

              float f_spring = k_pen * pen;
              float f_damp   = k_damp * vn;
              float f_mag    = f_spring - f_damp;
              if (f_mag < 0.0f) f_mag = 0.0f;

              fx = f_mag * n_x;
              fy = f_mag * n_y;
              fz = f_mag * n_z;

              if (mu_p > 0.0f) {
                float vt_x = vrel_x - vn*n_x;
                float vt_y = vrel_y - vn*n_y;
                float vt_z = vrel_z - vn*n_z;
                float vtMag2 = vt_x * vt_x + vt_y * vt_y + vt_z * vt_z;
                if (vtMag2 > 1e-16f) {
                  float invVt  = 1.0f / sqrtf(vtMag2);
                  float fFric  = mu_p * f_mag;
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

            ps.forceX[i] += fx;
            ps.forceY[i] += fy;
            ps.forceZ[i] += fz;

            #ifdef HAS_OPENMP
              #pragma omp atomic
            #endif
            ps.forceX[j] -= fx;
            #ifdef HAS_OPENMP
              #pragma omp atomic
            #endif
            ps.forceY[j] -= fy;
            #ifdef HAS_OPENMP
              #pragma omp atomic
            #endif
            ps.forceZ[j] -= fz;
          }
        }
      }
    }
  }
}

// ============================================================================
// Scalar kernel helpers — shared between SISD (full range) and SIMD (remainder)
// ============================================================================

/**
 * @brief Initialize snowpack particles in a sloped layer with random jitter.
 * @param sp      Snowpack to initialize (pos/radius/mass/wetness/alive).
 * @param start   Starting index for this kernel invocation (inclusive).
 * @param end     Ending index for this kernel invocation (exclusive).
 * @param params  SimParams providing slope geometry, spawn area, thickness, and particle properties.
 */
void initSnowpack_scalar(Snowpack &sp, int start, int end, const SimParams &params) {
  int N = sp.count;

  float sn = params.slopeSin;
  float cs = params.slopeCos;
  float H  = params.slopeHeight;
  float Ls = params.spawnLengthS;
  float Wz = params.spawnWidthZ;
  float S0 = params.spawnStartS;
  float Tn = params.spawnThicknessN;
  float pr = params.particleRadius;

  int layersN = 1;
  if (Tn > 0.0f) layersN = (std::max)(1, (int)(Tn / (2.0f * pr)));
  int perLayer = N / layersN;
  if (perLayer < 1) perLayer = 1;

  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int idx = start; idx < end; idx++) {
    uint32_t rngState = (rand() % 100 + 1) ^ (uint32_t)(idx * 2654435761u);

    int layer   = idx / perLayer;
    int inLayer = idx % perLayer;
    if (layer >= layersN) {
      layer = layersN - 1;
      inLayer = perLayer - 1;
    }

    int cols = (int)sqrtf((float)perLayer * Wz / Ls);
    if (cols < 1) cols = 1;

    int rows = (perLayer + cols - 1) / cols;
    int row = inLayer / cols;
    int col = inLayer % cols;
    if (row >= rows) row = rows - 1;

    float spacingS = Ls / (float)rows;
    float spacingZ = Wz / (float)cols;

    float s = S0 + row * spacingS + lcgRandom(rngState) * spacingS * 0.3f;
    float z = -Wz * 0.5f + col * spacingZ + lcgRandom(rngState) * spacingZ * 0.3f;
    float n = pr + layer * 2.0f * pr;

    float x0 = 0.0f;
    float y0 = H / cs;

    sp.posX[idx] = x0 + s * cs + n * sn;
    sp.posY[idx] = y0 - s * sn + n * cs;
    sp.posZ[idx] = z;

    sp.radius[idx]  = pr;
    sp.mass[idx]    = params.particleMass;

    sp.wetness[idx] = params.wetnessMin + lcgRandom(rngState) * (params.wetnessMax - params.wetnessMin);
    sp.alive[idx]   = AVAILABLE;
  }
}

/**
 * @brief Sort all snowpack SoA arrays in-place by posX ascending.
 *        Called once after initSnowpack. Enables per-frame binary search
 *        to limit captureFromSnowpack to a geometrically relevant window,
 *        mirroring the GPU range-limited scan (spOffset/spCount).
 * @param sp Snowpack to sort.
 */
void sortSnowpackByPosX(Snowpack &sp) {
  int N = sp.count;

  // Build sort permutation
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return sp.posX[a] < sp.posX[b];
  });

  // Apply permutation to every float SoA array
  std::vector<float> tmp(N);
  auto applyPerm = [&](float *arr) {
    for (int i = 0; i < N; i++) tmp[i] = arr[idx[i]];
    std::copy(tmp.begin(), tmp.end(), arr);
  };
  applyPerm(sp.posX);
  applyPerm(sp.posY);
  applyPerm(sp.posZ);
  applyPerm(sp.radius);
  applyPerm(sp.mass);
  applyPerm(sp.wetness);

  // Apply permutation to alive (enum, not float)
  std::vector<SnowpackParticleState> tmpState(N);
  for (int i = 0; i < N; i++) tmpState[i] = sp.alive[idx[i]];
  std::copy(tmpState.begin(), tmpState.end(), sp.alive);
}

/**
 * @brief Compute forces on shell particles from the snowball tether constraint.
 * @param shell   ParticleSystem representing the shell (reads pos/vel/attachLocal, writes force*).
 * @param ball    SnowballState representing the snowball (reads pos/vel/quaternion/radius).
 * @param start   Starting index for this kernel invocation (inclusive).
 * @param end     Ending index for this kernel invocation (exclusive).
 * @param params  SimParams providing simulation parameters (damping, gravity, tether constants).
 */
void forcesTether_scalar(ParticleSystem &shell, const SnowballState &ball, int start, int end, const SimParams &params) {
  float kd = params.damping;
  float g  = params.gravity;
  float K  = params.shellTetherK;
  float D  = params.shellTetherDamp;

  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int idx = start; idx < end; idx++) {
    float m = shell.mass[idx];

    float fx = -kd * m * shell.velX[idx];
    float fy = -m * g - kd * m * shell.velY[idx];
    float fz = -kd * m * shell.velZ[idx];

    float lx = shell.attachLocalX[idx];
    float ly = shell.attachLocalY[idx];
    float lz = shell.attachLocalZ[idx];

    float wx, wy, wz;
    quatRotateVec(ball.quatW, ball.quatX, ball.quatY, ball.quatZ, lx, ly, lz, wx, wy, wz);

    float pr    = shell.radius[idx];
    float tDist = ball.radius + pr;

    float tpx = ball.posX + wx * tDist;
    float tpy = ball.posY + wy * tDist;
    float tpz = ball.posZ + wz * tDist;

    float rx = wx * tDist;
    float ry = wy * tDist;
    float rz = wz * tDist;
    float tvx = ball.velX + (ball.omegaY * rz - ball.omegaZ * ry);
    float tvy = ball.velY + (ball.omegaZ * rx - ball.omegaX * rz);
    float tvz = ball.velZ + (ball.omegaX * ry - ball.omegaY * rx);

    float tfx = K * (tpx - shell.posX[idx]) + D * (tvx - shell.velX[idx]);
    float tfy = K * (tpy - shell.posY[idx]) + D * (tvy - shell.velY[idx]);
    float tfz = K * (tpz - shell.posZ[idx]) + D * (tvz - shell.velZ[idx]);

    fx += tfx;
    fy += tfy;
    fz += tfz;

    shell.forceX[idx] = fx;
    shell.forceY[idx] = fy;
    shell.forceZ[idx] = fz;
  }
}

/**
 * @brief Integrate shell particles with ground collision (slope) handling.
 * @param ps      ParticleSystem representing the shell (reads pos/vel/force, writes pos/vel).
 * @param start   Starting index for this kernel invocation (inclusive).
 * @param end     Ending index for this kernel invocation (exclusive).
 * @param params  SimParams providing time step, slope geometry, restitution, and friction.
 */
void integrateGround_scalar(ParticleSystem &ps, int start, int end, const SimParams &params) {
  float dt = params.dt;
  float e  = params.restitution;
  float mu = params.friction;

  float sn = params.slopeSin;
  float cs = params.slopeCos;
  float H  = params.slopeHeight;

  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int idx = start; idx < end; idx++) {
    if (ps.state[idx] != ACTIVE) continue;

    float inv_m = 1.0f / ps.mass[idx];

    float vx = ps.velX[idx] + ps.forceX[idx] * inv_m * dt;
    float vy = ps.velY[idx] + ps.forceY[idx] * inv_m * dt;
    float vz = ps.velZ[idx] + ps.forceZ[idx] * inv_m * dt;

    float px = ps.posX[idx] + vx * dt;
    float py = ps.posY[idx] + vy * dt;
    float pz = ps.posZ[idx] + vz * dt;

    float dist = px * sn + py * cs - H;

    float r = ps.radius[idx];
    if (dist < r) {
      float pen = r - dist;
      px += pen * sn;
      py += pen * cs;

      float vn = vx * sn + vy * cs;

      if (vn < 0.0f) {
        float dvn = -(1.0f + e) * vn;

        float vtX = vx - vn * sn;
        float vtY = vy - vn * cs;
        float vtZ = vz;

        float vtMag = std::sqrt(vtX * vtX + vtY * vtY + vtZ * vtZ);
        float fricImp = mu * std::abs(dvn);

        if (vtMag > 1e-8f && fricImp < vtMag) {
          float scale = 1.0f - fricImp / vtMag;
          vtX *= scale;
          vtY *= scale;
          vtZ *= scale;
        } else if (vtMag <= 1e-8f) {
          // No tangential motion
        } else {
          vtX = 0.0f;
          vtY = 0.0f;
          vtZ = 0.0f;
        }

        vx = vtX + (vn + dvn) * sn;
        vy = vtY + (vn + dvn) * cs;
        vz = vtZ;
      }
    }

    ps.posX[idx] = px;
    ps.posY[idx] = py;
    ps.posZ[idx] = pz;

    ps.velX[idx] = vx;
    ps.velY[idx] = vy;
    ps.velZ[idx] = vz;
  }
}

/**
 * @brief Compute grid cell hashes for shell particles and store particle indices.
 * @param grid    GridData to fill (cellHash and particleIndex).
 * @param ps      ParticleSystem representing the shell (reads pos, writes cellHash and particleIndex).
 * @param start   Starting index for this kernel invocation (inclusive).
 * @param end     Ending index for this kernel invocation (exclusive).
 */
void buildGrid_hashScalar(GridData &grid, const ParticleSystem &ps, int start, int end) {
  for (int idx = start; idx < end; idx++) {
    int cx = (int)floorf((ps.posX[idx] - grid.originX) / grid.cellSize);
    int cy = (int)floorf((ps.posY[idx] - grid.originY) / grid.cellSize);
    int cz = (int)floorf((ps.posZ[idx] - grid.originZ) / grid.cellSize);

    cx = (std::max)(0, (std::min)(cx, grid.gridDimX - 1));
    cy = (std::max)(0, (std::min)(cy, grid.gridDimY - 1));
    cz = (std::max)(0, (std::min)(cz, grid.gridDimZ - 1));

    grid.cellHash[idx]      = (uint32_t)(cz * grid.gridDimX * grid.gridDimY + cy * grid.gridDimX + cx);
    grid.particleIndex[idx] = (uint32_t)idx;
  }
}

/**
 * @brief Sort particles by cell hash and compute cell start/end indices.
 * @param grid    GridData with cellHash and particleIndex filled, to be sorted and have cellStart/cellEnd computed.
 * @param shellN  Number of shell particles (size of cellHash and particleIndex arrays).
 */
void buildGrid_sortAndRanges(GridData &grid, int shellN) {
  std::sort(grid.particleIndex, grid.particleIndex + shellN,
    [&](uint32_t a, uint32_t b) {
      return grid.cellHash[a] < grid.cellHash[b];
    });

  uint32_t *sortedHash = (uint32_t *)ALLOC(shellN * sizeof(uint32_t));
  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int i = 0; i < shellN; i++) {
    sortedHash[i] = grid.cellHash[grid.particleIndex[i]];
  }
  std::copy_n(sortedHash, shellN, grid.cellHash);
  FREE(sortedHash);

  #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
  #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int idx = 0; idx < shellN; idx++) {
    uint32_t hash = grid.cellHash[idx];

    if (idx == 0 || hash != grid.cellHash[idx - 1])
      grid.cellStart[hash] = idx;

    if (idx == shellN - 1 || hash != grid.cellHash[idx + 1])
      grid.cellEnd[hash] = idx + 1;
  }
}

/**
 * @brief Build the spatial grid for shell particles: compute cell hashes, sort particles, and compute cell ranges.
 *        Uses a static variable to track previously occupied cells for sparse clearing.
 * @param grid    GridData to fill and maintain.
 * @param ps      ParticleSystem representing the shell (reads pos, writes cellHash and particleIndex).
 * @param shellN  Number of shell particles (size of cellHash and particleIndex arrays).
 */
void buildGrid_scalar(GridData &grid, const ParticleSystem &ps, int shellN) {
  static int prevN = 0;

  // Sparse clear: reset only cells occupied last frame.
  if (prevN > 0) {
    for (int i = 0; i < prevN; i++) {
      uint32_t hash = grid.cellHash[i];
      if (i == 0 || hash != grid.cellHash[i - 1]) {
        grid.cellStart[hash] = INACTIVE;
        grid.cellEnd[hash]   = INACTIVE;
      }
    }
  }

  buildGrid_hashScalar(grid, ps, 0, shellN);
  buildGrid_sortAndRanges(grid, shellN);

  prevN = shellN;
}

/**
 * @brief Capture snowpack particles into the shell based on proximity to the snowball and stickiness probability.
 *        Updates the snowpack particle states to CONSUMED and adds new shell particles at the attachment points on the snowball surface.
 * @param sp            Snowpack containing particles to potentially capture (reads pos/radius/mass/wetness/alive, writes alive).
 * @param shell         ParticleSystem representing the shell (writes new particles at shellN + offset).
 * @param shellN        Current number of shell particles (used to compute new particle indices).
 * @param ball          SnowballState representing the snowball (reads pos/vel/quaternion/radius).
 * @param frame         Current frame number (used for RNG seeding).
 * @param start         Starting index for this kernel invocation (inclusive).
 * @param end           Ending index for this kernel invocation (exclusive).
 * @param captureCount  Reference to the count of captured particles so far in this frame (used to compute slot offsets and check max capture per frame).
 * @param captureMass   Reference to the total mass captured so far in this frame (updated with the mass of newly captured particles).
 * @param params        SimParams providing stickiness parameters and max capture per frame.
 */
void captureFromSnowpack_scalar(
    Snowpack &sp,
    ParticleSystem &shell, int shellN,
    const SnowballState &ball,
    int frame,
    int start, int end,
    int &captureCount, float &captureMass,
    const SimParams &params)
{
  float ballRadius = ball.radius;

  // Captures already accumulated before this call (e.g. from the SIMD phase).
  // Used to compute the correct shell slot offset and check the per-frame cap.
  const int slotBase = captureCount;

  int   localCount = 0;
  float localMass  = 0.0f;

  // NOTE: OpenMP disabled here — reduction(+:localCount) gives each thread a
  // private copy starting at 0, so multiple threads would compute the same
  // slot = slotBase + 0, 1, … and write to the same shell indices (data race).
  // Also, 'break' is illegal inside a #pragma omp parallel for.
  // #ifdef HAS_OPENMP
  //   #pragma omp parallel for schedule(static) reduction(+:localCount, localMass)
  // #endif
  #ifndef SIMD
    #pragma loop(no_vector) // disable auto-vectorization
  #endif
  for (int idx = start; idx < end; idx++) {
    if (sp.alive[idx] != AVAILABLE) continue;

    float px  = sp.posX[idx];
    float py  = sp.posY[idx];
    float pz  = sp.posZ[idx];
    float spr = sp.radius[idx];

    float dx    = px - ball.posX;
    float dy    = py - ball.posY;
    float dz    = pz - ball.posZ;
    float dist2 = dx*dx + dy*dy + dz*dz;

    float captureR = ballRadius + spr + 0.3f;
    if (dist2 > captureR * captureR) continue;

    float dist = sqrtf(dist2);
    if (dist < 1e-8f) continue;

    float vrelX   = -ball.velX;
    float vrelY   = -ball.velY;
    float vrelZ   = -ball.velZ;
    float vrelMag = sqrtf(vrelX*vrelX + vrelY*vrelY + vrelZ*vrelZ);

    float w      = sp.wetness[idx];
    float logit  = params.stickK0
                 + params.stickK1 * w
                 - params.stickK2 * vrelMag
                 + params.stickRadiusBoost * ballRadius;
    float clamp  = params.logitClamp;
    logit = fminf(fmaxf(logit, -clamp), clamp);
    float P_stick = 1.0f / (1.0f + expf(-logit));

    uint32_t h = (uint32_t)idx ^ (uint32_t)(frame * 2654435761u);
    h ^= h >> 16; h *= 0x45d9f3bu;
    h ^= h >> 16; h *= 0x45d9f3bu;
    h ^= h >> 16;
    float rnd = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

    if (rnd >= P_stick) continue;

    int slot = slotBase + localCount;
    if (params.maxCapturePerFrm > 0 && slot >= params.maxCapturePerFrm) break;
    if (shellN + slot >= shell.capacity) break;

    sp.alive[idx]  = CONSUMED;
    localMass     += sp.mass[idx];

    int si = shellN + slot;

    float invDist = 1.0f / dist;
    float nx = dx * invDist;
    float ny = dy * invDist;
    float nz = dz * invDist;

    float surfDist    = ballRadius + spr;
    shell.posX[si]    = ball.posX + nx * surfDist;
    shell.posY[si]    = ball.posY + ny * surfDist;
    shell.posZ[si]    = ball.posZ + nz * surfDist;

    float rx = nx * surfDist;
    float ry = ny * surfDist;
    float rz = nz * surfDist;
    shell.velX[si] = ball.velX + (ball.omegaY * rz - ball.omegaZ * ry);
    shell.velY[si] = ball.velY + (ball.omegaZ * rx - ball.omegaX * rz);
    shell.velZ[si] = ball.velZ + (ball.omegaX * ry - ball.omegaY * rx);

    shell.forceX[si]  = 0.0f;
    shell.forceY[si]  = 0.0f;
    shell.forceZ[si]  = 0.0f;
    shell.mass[si]    = sp.mass[idx];
    shell.radius[si]  = spr;
    shell.state[si]   = ACTIVE;
    shell.wetness[si] = w;

    quatInvRotateVec(ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
                     nx, ny, nz,
                     shell.attachLocalX[si], shell.attachLocalY[si], shell.attachLocalZ[si]);

    localCount++;
  }

  captureCount += localCount;
  captureMass  += localMass;
}

// ============================================================================
// runSimulation - full simulation loop
// ============================================================================
/**
 * @brief Run the full simulation loop.
 * @param params     Simulation parameters (may be modified for slope precomputation).
 * @return Exit code (0 = success).
 */
int runSimulation(SimParams &params) {
  int shellN = 0;

  #ifdef SIMD
    #ifdef HAS_OPENMP
      printf("[OpenMP] %d threads available\n", omp_get_max_threads());
    #else
      printf("[OpenMP] disabled (single-threaded)\n");
    #endif
  #endif
  
  Logger logger;
  KernelTimings  kt = {};
  ProfilingStats profStats;
  profStats.reset();

  #ifdef SIMD
    if (params.logInterval > 0) logger.openCSV((timestamp + "_log_SIMD_cpu.csv").c_str());
  #else
    if (params.logInterval > 0) logger.openCSV((timestamp + "_log_SISD_cpu.csv").c_str());
  #endif
  
  // -------------------------------------------------------------------
  // Allocate data structures
  // -------------------------------------------------------------------
  Snowpack snowpack;
  allocateSnowpack(snowpack, params.numParticles);
  initSnowpack(snowpack, params);
  sortSnowpackByPosX(snowpack);   // sort by posX once for per-frame binary search
  
  SnowballState ball;
  initSnowball(ball, params);

  ParticleSystem shell;
  int shellCap = 256;  // initial small capacity, grows dynamically
  allocateParticleSystem(shell, shellCap);

  GridData grid;
  allocateGrid(grid, params, shell.capacity);

  // -------------------------------------------------------------------
  // Simulation loop
  // -------------------------------------------------------------------
  auto wallStart = std::chrono::high_resolution_clock::now();
  int totalFrames = 0;  // count how many frames ran
  for (int frame = 0; ; frame++) {
    auto frameStart = std::chrono::high_resolution_clock::now();

    // 1. Capture from snowpack
    auto t0 = std::chrono::high_resolution_clock::now();
    int newCount = 0;
    float capturedMass = 0.0f;

    // Ensure capacity before capture — grow if needed
    {
      int need = params.maxCapturePerFrm > 0 ? params.maxCapturePerFrm : (params.numParticles - shellN);
      if (need > 0 && shellN + need > shell.capacity) {
        int newCap = shell.capacity > 0 ? shell.capacity : 256;
        while (newCap < shellN + need) newCap *= 2;
        growParticleSystem(shell, shellN, newCap);
        growGrid(grid, newCap);
      }
    }
    // Range-limited capture: binary search on sorted posX to skip irrelevant particles
    float captureR = ball.radius + params.particleRadius + 0.3f;
    int spLo = (int)(std::lower_bound(snowpack.posX, snowpack.posX + snowpack.count,
                                      ball.posX - captureR) - snowpack.posX);
    int spHi = (int)(std::upper_bound(snowpack.posX, snowpack.posX + snowpack.count,
                                      ball.posX + captureR) - snowpack.posX);
    newCount = captureFromSnowpack(snowpack, shell, shellN, ball, params, &capturedMass, frame, spLo, spHi - spLo);

    int maxCapWrite = params.maxCapturePerFrm > 0 ? params.maxCapturePerFrm : newCount;
    int actualNew = (std::min)(newCount, (std::min)(maxCapWrite, shell.capacity - shellN));
    shellN += actualNew;
    auto t1 = std::chrono::high_resolution_clock::now();
    kt.captureMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 2. Update core mass / radius (momentum conservation)
    updateCoreMassAfterCapture(ball, actualNew, capturedMass, params.snowDensity);

    // 3. Forces + Tether (fused into one pass)
    t0 = std::chrono::high_resolution_clock::now();
    forcesTether(shell, ball, shellN, params);
    t1 = std::chrono::high_resolution_clock::now();
    kt.shellForcesMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 4. Grid build
    t0 = std::chrono::high_resolution_clock::now();
    if (shellN > 0) buildGrid(grid, shell, shellN);
    t1 = std::chrono::high_resolution_clock::now();
    kt.shellGridMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 5. Neighbor collision
    t0 = std::chrono::high_resolution_clock::now();
    if (shellN > 0) neighborCollision(shell, grid, shellN, params);
    t1 = std::chrono::high_resolution_clock::now();
    kt.shellCollisionMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 6. Integrate + Ground Collision (fused into one pass)
    t0 = std::chrono::high_resolution_clock::now();
    integrateGround(shell, shellN, params);
    t1 = std::chrono::high_resolution_clock::now();
    kt.shellIntegrateMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 7. Core rigid body update
    t0 = std::chrono::high_resolution_clock::now();
    updateSnowball(ball, params);
    t1 = std::chrono::high_resolution_clock::now();
    kt.coreUpdateMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    auto frameEnd = std::chrono::high_resolution_clock::now();
    kt.totalFrameMs = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();

    profStats.accumulate(kt);

    // Check if ball has reached the ground
    bool ballAtGround = (ball.posY <= ball.radius + 0.01f) && (ball.velY >= -0.01f);
    totalFrames = frame + 1;

    // -------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------
    if ((params.logInterval > 0) &&
        (frame % params.logInterval == 0 || ballAtGround)) {

      FrameStats stats = {};
      stats.frame               = frame;
      stats.shellCount          = shellN;
      stats.snowpackAlive       = snowpack.count - ball.capturedParticleCount;
      stats.ballMass            = ball.mass;
      stats.ballRadius          = ball.radius;
      stats.ballPosX            = ball.posX;
      stats.ballPosY            = ball.posY;
      stats.ballPosZ            = ball.posZ;
      stats.ballVelX            = ball.velX;
      stats.ballVelY            = ball.velY;
      stats.ballVelZ            = ball.velZ;
      stats.frameTimeMs         = kt.totalFrameMs;
      stats.captureMs           = kt.captureMs;
      stats.shellForcesTetherMs = kt.shellForcesMs;
      stats.shellGridMs         = kt.shellGridMs;
      stats.shellCollisionMs    = kt.shellCollisionMs;
      stats.shellIntGroundMs    = kt.shellIntegrateMs;
      stats.coreUpdateMs        = kt.coreUpdateMs;

      logger.logFrame(stats);
      if (shellN > 0) printGridStats(grid, shellN, frame);
    }

    // -------------------------------------------------------------------
    // Trace snapshot (shell particles)
    // -------------------------------------------------------------------
    if (params.traceInterval > 0 && (frame % params.traceInterval == 0) && shellN > 0) {
      char fname[64];
      #ifdef SIMD
        snprintf(fname, sizeof(fname), "%s_%05d_trace_SIMD_cpu.csv", timestamp.c_str(), frame);
      #else
        snprintf(fname, sizeof(fname), "%s_%05d_trace_SISD_cpu.csv", timestamp.c_str(), frame);
      #endif
      FILE *tf = fopen(fname, "w");
      if (tf) {
          fprintf(tf, "posX,posY,posZ,velX,velY,velZ,forceX,forceY,forceZ,attachX,attachY,attachZ,mass,radius,state,wetness\n");
          for (int i = 0; i < shellN; i++) {
            fprintf(tf, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f\n",
                    shell.posX[i], shell.posY[i], shell.posZ[i],
                    shell.velX[i], shell.velY[i], shell.velZ[i],
                    shell.forceX[i], shell.forceY[i], shell.forceZ[i],
                    shell.attachLocalX[i], shell.attachLocalY[i], shell.attachLocalZ[i],
                    shell.mass[i],
                    shell.radius[i],
                    shell.state[i],
                    shell.wetness[i]);
          }
          fclose(tf);
          printf("  [Trace] wrote %s (%d shell particles)\n", fname, shellN);
        }
    }

    if (ballAtGround) break;
  }

  auto wallEnd = std::chrono::high_resolution_clock::now();
  double wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
  double wallFps = (wallMs > 0.0) ? totalFrames * 1000.0 / wallMs : 0.0;

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  printf("Wall-clock: %.1f ms total, %.1f FPS (%.3f ms/frame)\n",
          wallMs, wallFps, (totalFrames > 0) ? wallMs / totalFrames : 0.0);
  printf("[CPU] Simulation completed: %d frames\n", totalFrames);
  printSimulationSummary(ball, shellN);
  profStats.printSummary(shellN, snowpack.count, totalFrames);

  // -------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------
  logger.close();
  freeGrid(grid);
  freeParticleSystem(shell);
  freeSnowpack(snowpack);

  return 0;
}