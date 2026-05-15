#include "../../shared/include/global.h"
#include "../../shared/include/simulation_CPU.h"

#include <cmath> // mandatory on LINUX

// SIMD headers selected by host architecture.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>   // SSE / SSE2 intrinsics
  #define CPU_SIMD_X86 1
  #define CPU_SIMD_NEON 0
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
  #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define CPU_SIMD_NEON 1
  #else
    #define CPU_SIMD_NEON 0
  #endif
  #define CPU_SIMD_X86 0
#else
  #define CPU_SIMD_X86 0
  #define CPU_SIMD_NEON 0
#endif

#ifdef HAS_OPENMP
  #include <omp.h>
#endif

// ============================================================================
// initSnowpack - place particles on slope surface with jitter and wetness
// ============================================================================
/**
 * @brief Initialize snowpack particles on the slope surface.
 * @param sp      Snowpack to populate.
 * @param params  SimParams providing slope geometry and particle properties.
 */
void initSnowpack(Snowpack &sp, const SimParams &params) {
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

  #if CPU_SIMD_X86
    // --- SSE path: 4 particles per iteration ----------------------------------
    const __m128 vSn = _mm_set1_ps(sn);
    const __m128 vCs = _mm_set1_ps(cs);
    const __m128 vS0 = _mm_set1_ps(S0);
    const __m128 vPr = _mm_set1_ps(pr);
    const __m128 vTwoPr = _mm_set1_ps(2.0f * pr);
    const __m128 vHalfWz = _mm_set1_ps(-Wz * 0.5f);
    const __m128 vY0 = _mm_set1_ps(H / cs);

    const __m128 vMass = _mm_set1_ps(params.particleMass);
    const __m128 vWmin = _mm_set1_ps(params.wetnessMin);
    const __m128 vWdiff = _mm_set1_ps(params.wetnessMax - params.wetnessMin);

    const int N4 = N & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float s_arr[4];
      float z_arr[4];
      float n_arr[4];
      float wet_arr[4];
      
      for (int k = 0; k < 4; k++) {
        int idx = i + k;
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

        s_arr[k] = s;
        z_arr[k] = z;
        n_arr[k] = n;

        wet_arr[k] = params.wetnessMin + lcgRandom(rngState) * (params.wetnessMax - params.wetnessMin);
      }

      __m128 s = _mm_loadu_ps(s_arr);
      __m128 z = _mm_loadu_ps(z_arr);
      __m128 n = _mm_loadu_ps(n_arr);
      __m128 wet = _mm_loadu_ps(wet_arr);

      // world transform
      __m128 x = _mm_add_ps(_mm_mul_ps(s, vCs), _mm_mul_ps(n, vSn));
      __m128 y = _mm_sub_ps(vY0, _mm_mul_ps(s, vSn));
      y = _mm_add_ps(y, _mm_mul_ps(n, vCs));

      _mm_storeu_ps(&sp.posX[i], x);
      _mm_storeu_ps(&sp.posY[i], y);
      _mm_storeu_ps(&sp.posZ[i], z);

      _mm_storeu_ps(&sp.radius[i], vPr);
      _mm_storeu_ps(&sp.mass[i], vMass);
      _mm_storeu_ps(&sp.wetness[i], wet);

      for (int k = 0; k < 4; k++) {
        sp.alive[i + k] = AVAILABLE;
      }    }

    int scalarStart = N4;
  #elif CPU_SIMD_NEON
    // --- NEON path: 4 particles per iteration ---------------------------------
    const float32x4_t vSn = vdupq_n_f32(sn);
    const float32x4_t vCs = vdupq_n_f32(cs);
    const float32x4_t vY0 = vdupq_n_f32(H / cs);
    const float32x4_t vPr = vdupq_n_f32(pr);
    const float32x4_t vMass = vdupq_n_f32(params.particleMass);

    const int N4 = N & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float s_arr[4];
      float z_arr[4];
      float n_arr[4];
      float wet_arr[4];

      for (int k = 0; k < 4; k++) {
        int idx = i + k;
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

        s_arr[k] = s;
        z_arr[k] = z;
        n_arr[k] = n;

        wet_arr[k] = params.wetnessMin + lcgRandom(rngState) * (params.wetnessMax - params.wetnessMin);
      }

      float32x4_t s = vld1q_f32(s_arr);
      float32x4_t z = vld1q_f32(z_arr);
      float32x4_t n = vld1q_f32(n_arr);
      float32x4_t wet = vld1q_f32(wet_arr);

      float32x4_t x = vaddq_f32(vmulq_f32(s, vCs), vmulq_f32(n, vSn));
      float32x4_t y = vaddq_f32(vsubq_f32(vY0, vmulq_f32(s, vSn)), vmulq_f32(n, vCs));

      vst1q_f32(&sp.posX[i], x);
      vst1q_f32(&sp.posY[i], y);
      vst1q_f32(&sp.posZ[i], z);

      vst1q_f32(&sp.radius[i], vPr);
      vst1q_f32(&sp.mass[i], vMass);
      vst1q_f32(&sp.wetness[i], wet);

      for (int k = 0; k < 4; k++) {
        sp.alive[i + k] = AVAILABLE;
      }
    }

    int scalarStart = N4;
  #else
    int scalarStart = 0;
  #endif

  initSnowpack_scalar(sp, scalarStart, N, params);
}

// ============================================================================
// forcesTether - FUSED: gravity + damping + shell-core tether spring
// Combines applyForces + shellCoreTether into one pass to reduce
// memory round-trips (setup forces, then accumulate tether).
// ============================================================================
/**
 * @brief Fused forces + tether: gravity + damping, then shell-particle tether.
 * @param shell   Shell ParticleSystem (reads attachLocal*, writes force*).
 * @param ball    Current snowball state.
 * @param shellN  Number of shell particles.
 * @param params  SimParams.
 */
void forcesTether(ParticleSystem &shell, const SnowballState &ball, int shellN, const SimParams &params) {      
  float kd = params.damping;
  float g  = params.gravity;

  // --- Tether spring-damper ---
  float K  = params.shellTetherK;
  float D  = params.shellTetherDamp;

  #if CPU_SIMD_X86
    // --- SSE path: 4 particles per iteration ----------------------------------
    const __m128 vNegKd = _mm_set1_ps(-kd);
    const __m128 vNegG  = _mm_set1_ps(-g);
    const __m128 vK     = _mm_set1_ps(K);
    const __m128 vD     = _mm_set1_ps(D);
    const __m128 vTwo   = _mm_set1_ps(2.0f);

    const __m128 vBpx = _mm_set1_ps(ball.posX);
    const __m128 vBpy = _mm_set1_ps(ball.posY);
    const __m128 vBpz = _mm_set1_ps(ball.posZ);
    const __m128 vBvx = _mm_set1_ps(ball.velX);
    const __m128 vBvy = _mm_set1_ps(ball.velY);
    const __m128 vBvz = _mm_set1_ps(ball.velZ);
    const __m128 vBR  = _mm_set1_ps(ball.radius);
    const __m128 vOx  = _mm_set1_ps(ball.omegaX);
    const __m128 vOy  = _mm_set1_ps(ball.omegaY);
    const __m128 vOz  = _mm_set1_ps(ball.omegaZ);
    const __m128 vQw  = _mm_set1_ps(ball.quatW);
    const __m128 vQx  = _mm_set1_ps(ball.quatX);
    const __m128 vQy  = _mm_set1_ps(ball.quatY);
    const __m128 vQz  = _mm_set1_ps(ball.quatZ);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      __m128 m  = _mm_load_ps(&shell.mass[i]);
      __m128 vx = _mm_load_ps(&shell.velX[i]);
      __m128 vy = _mm_load_ps(&shell.velY[i]);
      __m128 vz = _mm_load_ps(&shell.velZ[i]);

      // PHASE 1: applyForces - reset forces
      __m128 kdm = _mm_mul_ps(vNegKd, m);                                      // -kd * m
      __m128 fx = _mm_mul_ps(kdm, vx);                                         // forceX = -kd * m * velX
      __m128 fy = _mm_add_ps(_mm_mul_ps(m, vNegG), _mm_mul_ps(kdm, vy));      // forceY = -m*g - kd*m*velY
      __m128 fz = _mm_mul_ps(kdm, vz);                                         // forceZ = -kd * m * velZ

      // PHASE 2: shellCoreTether - accumulate tether forces
      // Load local attachment vectors
      __m128 lx = _mm_load_ps(&shell.attachLocalX[i]);
      __m128 ly = _mm_load_ps(&shell.attachLocalY[i]);
      __m128 lz = _mm_load_ps(&shell.attachLocalZ[i]);

      // Quaternion rotation: q * v * q^{-1}
      __m128 tx = _mm_mul_ps(vTwo, _mm_sub_ps(_mm_mul_ps(vQy, lz), _mm_mul_ps(vQz, ly)));
      __m128 ty = _mm_mul_ps(vTwo, _mm_sub_ps(_mm_mul_ps(vQz, lx), _mm_mul_ps(vQx, lz)));
      __m128 tz = _mm_mul_ps(vTwo, _mm_sub_ps(_mm_mul_ps(vQx, ly), _mm_mul_ps(vQy, lx)));

      __m128 wx = _mm_add_ps(lx, _mm_add_ps(_mm_mul_ps(vQw, tx), _mm_sub_ps(_mm_mul_ps(vQy, tz), _mm_mul_ps(vQz, ty))));
      __m128 wy = _mm_add_ps(ly, _mm_add_ps(_mm_mul_ps(vQw, ty), _mm_sub_ps(_mm_mul_ps(vQz, tx), _mm_mul_ps(vQx, tz))));
      __m128 wz = _mm_add_ps(lz, _mm_add_ps(_mm_mul_ps(vQw, tz), _mm_sub_ps(_mm_mul_ps(vQx, ty), _mm_mul_ps(vQy, tx))));

      __m128 pr    = _mm_load_ps(&shell.radius[i]);
      __m128 tDist = _mm_add_ps(vBR, pr);

      // Tether target position
      __m128 tpx = _mm_add_ps(vBpx, _mm_mul_ps(wx, tDist));
      __m128 tpy = _mm_add_ps(vBpy, _mm_mul_ps(wy, tDist));
      __m128 tpz = _mm_add_ps(vBpz, _mm_mul_ps(wz, tDist));

      // Tether target velocity
      __m128 rx = _mm_mul_ps(wx, tDist);
      __m128 ry = _mm_mul_ps(wy, tDist);
      __m128 rz = _mm_mul_ps(wz, tDist);

      __m128 tvx = _mm_add_ps(vBvx, _mm_sub_ps(_mm_mul_ps(vOy, rz), _mm_mul_ps(vOz, ry)));
      __m128 tvy = _mm_add_ps(vBvy, _mm_sub_ps(_mm_mul_ps(vOz, rx), _mm_mul_ps(vOx, rz)));
      __m128 tvz = _mm_add_ps(vBvz, _mm_sub_ps(_mm_mul_ps(vOx, ry), _mm_mul_ps(vOy, rx)));

      // Spring-damper tether
      __m128 spx = _mm_load_ps(&shell.posX[i]);
      __m128 spy = _mm_load_ps(&shell.posY[i]);
      __m128 spz = _mm_load_ps(&shell.posZ[i]);

      __m128 tfx = _mm_add_ps(_mm_mul_ps(vK, _mm_sub_ps(tpx, spx)), _mm_mul_ps(vD, _mm_sub_ps(tvx, vx)));
      __m128 tfy = _mm_add_ps(_mm_mul_ps(vK, _mm_sub_ps(tpy, spy)), _mm_mul_ps(vD, _mm_sub_ps(tvy, vy)));
      __m128 tfz = _mm_add_ps(_mm_mul_ps(vK, _mm_sub_ps(tpz, spz)), _mm_mul_ps(vD, _mm_sub_ps(tvz, vz)));

      // Accumulate
      fx = _mm_add_ps(fx, tfx);
      fy = _mm_add_ps(fy, tfy);
      fz = _mm_add_ps(fz, tfz);

      _mm_store_ps(&shell.forceX[i], fx);
      _mm_store_ps(&shell.forceY[i], fy);
      _mm_store_ps(&shell.forceZ[i], fz);
    }

    int scalarStart = N4;
  #elif CPU_SIMD_NEON

    // --- NEON path: 4 particles per iteration ---------------------------------
    const float32x4_t vNegKd = vdupq_n_f32(-kd);
    const float32x4_t vNegG  = vdupq_n_f32(-g);
    const float32x4_t vK     = vdupq_n_f32(K);
    const float32x4_t vD     = vdupq_n_f32(D);
    const float32x4_t vTwo   = vdupq_n_f32(2.0f);

    const float32x4_t vBpx = vdupq_n_f32(ball.posX);
    const float32x4_t vBpy = vdupq_n_f32(ball.posY);
    const float32x4_t vBpz = vdupq_n_f32(ball.posZ);
    const float32x4_t vBvx = vdupq_n_f32(ball.velX);
    const float32x4_t vBvy = vdupq_n_f32(ball.velY);
    const float32x4_t vBvz = vdupq_n_f32(ball.velZ);
    const float32x4_t vBR  = vdupq_n_f32(ball.radius);
    const float32x4_t vOx  = vdupq_n_f32(ball.omegaX);
    const float32x4_t vOy  = vdupq_n_f32(ball.omegaY);
    const float32x4_t vOz  = vdupq_n_f32(ball.omegaZ);
    const float32x4_t vQw  = vdupq_n_f32(ball.quatW);
    const float32x4_t vQx  = vdupq_n_f32(ball.quatX);
    const float32x4_t vQy  = vdupq_n_f32(ball.quatY);
    const float32x4_t vQz  = vdupq_n_f32(ball.quatZ);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float32x4_t m  = vld1q_f32(&shell.mass[i]);
      float32x4_t vx = vld1q_f32(&shell.velX[i]);
      float32x4_t vy = vld1q_f32(&shell.velY[i]);
      float32x4_t vz = vld1q_f32(&shell.velZ[i]);

      // PHASE 1: applyForces - reset forces
      float32x4_t kdm = vmulq_f32(vNegKd, m);                                  // -kd * m
      float32x4_t fx = vmulq_f32(kdm, vx);
      float32x4_t fy = vaddq_f32(vmulq_f32(m, vNegG), vmulq_f32(kdm, vy));
      float32x4_t fz = vmulq_f32(kdm, vz);

      // PHASE 2: shellCoreTether - accumulate tether forces
      float32x4_t lx = vld1q_f32(&shell.attachLocalX[i]);
      float32x4_t ly = vld1q_f32(&shell.attachLocalY[i]);
      float32x4_t lz = vld1q_f32(&shell.attachLocalZ[i]);

      float32x4_t tx = vmulq_f32(vTwo, vsubq_f32(vmulq_f32(vQy, lz), vmulq_f32(vQz, ly)));
      float32x4_t ty = vmulq_f32(vTwo, vsubq_f32(vmulq_f32(vQz, lx), vmulq_f32(vQx, lz)));
      float32x4_t tz = vmulq_f32(vTwo, vsubq_f32(vmulq_f32(vQx, ly), vmulq_f32(vQy, lx)));

      float32x4_t wx = vaddq_f32(lx, vaddq_f32(vmulq_f32(vQw, tx), vsubq_f32(vmulq_f32(vQy, tz), vmulq_f32(vQz, ty))));
      float32x4_t wy = vaddq_f32(ly, vaddq_f32(vmulq_f32(vQw, ty), vsubq_f32(vmulq_f32(vQz, tx), vmulq_f32(vQx, tz))));
      float32x4_t wz = vaddq_f32(lz, vaddq_f32(vmulq_f32(vQw, tz), vsubq_f32(vmulq_f32(vQx, ty), vmulq_f32(vQy, tx))));

      float32x4_t pr    = vld1q_f32(&shell.radius[i]);
      float32x4_t tDist = vaddq_f32(vBR, pr);

      float32x4_t tpx = vaddq_f32(vBpx, vmulq_f32(wx, tDist));
      float32x4_t tpy = vaddq_f32(vBpy, vmulq_f32(wy, tDist));
      float32x4_t tpz = vaddq_f32(vBpz, vmulq_f32(wz, tDist));

      float32x4_t rx = vmulq_f32(wx, tDist);
      float32x4_t ry = vmulq_f32(wy, tDist);
      float32x4_t rz = vmulq_f32(wz, tDist);

      float32x4_t tvx = vaddq_f32(vBvx, vsubq_f32(vmulq_f32(vOy, rz), vmulq_f32(vOz, ry)));
      float32x4_t tvy = vaddq_f32(vBvy, vsubq_f32(vmulq_f32(vOz, rx), vmulq_f32(vOx, rz)));
      float32x4_t tvz = vaddq_f32(vBvz, vsubq_f32(vmulq_f32(vOx, ry), vmulq_f32(vOy, rx)));

      float32x4_t spx = vld1q_f32(&shell.posX[i]);
      float32x4_t spy = vld1q_f32(&shell.posY[i]);
      float32x4_t spz = vld1q_f32(&shell.posZ[i]);

      float32x4_t tfx = vaddq_f32(vmulq_f32(vK, vsubq_f32(tpx, spx)), vmulq_f32(vD, vsubq_f32(tvx, vx)));
      float32x4_t tfy = vaddq_f32(vmulq_f32(vK, vsubq_f32(tpy, spy)), vmulq_f32(vD, vsubq_f32(tvy, vy)));
      float32x4_t tfz = vaddq_f32(vmulq_f32(vK, vsubq_f32(tpz, spz)), vmulq_f32(vD, vsubq_f32(tvz, vz)));

      fx = vaddq_f32(fx, tfx);
      fy = vaddq_f32(fy, tfy);
      fz = vaddq_f32(fz, tfz);

      vst1q_f32(&shell.forceX[i], fx);
      vst1q_f32(&shell.forceY[i], fy);
      vst1q_f32(&shell.forceZ[i], fz);
    }

    int scalarStart = N4;
  #else
    int scalarStart = 0;
  #endif
    
  // --- Remainder loop for any particles past the last multiple of 4 ---
  forcesTether_scalar(shell, ball, scalarStart, shellN, params);
}

// ============================================================================
// buildGrid - hash + sort + cell ranges
// ============================================================================
/**
 * @brief Build spatial hash grid: compute hashes, sort, build cell ranges.
 * @param grid   GridData to populate (cellHash, particleIndex, cellStart/End).
 * @param ps     ParticleSystem providing particle positions.
 * @param shellN Number of shell particles.
 */
void buildGrid(GridData &grid, const ParticleSystem &ps, int shellN) {
  // Sparse clear: reset only the cells that were occupied in the previous
  // frame.  grid.cellHash still holds sorted hashes from the last call,
  // so we walk them and set cellStart/cellEnd = INACTIVE for each unique hash.
  static int prevN = 0;
  if (prevN > 0) {
    #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < prevN; i++) {
      uint32_t hash = grid.cellHash[i];
      if (i == 0 || hash != grid.cellHash[i - 1]) {
        grid.cellStart[hash] = INACTIVE;
        grid.cellEnd[hash]   = INACTIVE;
      }
    }
  }

  #if CPU_SIMD_X86
    // --- SSE path: 4 particles per iteration ----------------------------------
    const __m128 vOriginX = _mm_set1_ps(grid.originX);
    const __m128 vOriginY = _mm_set1_ps(grid.originY);
    const __m128 vOriginZ = _mm_set1_ps(grid.originZ);
    const __m128 vInvCell = _mm_set1_ps(1.0f / grid.cellSize);

    const __m128i vZero = _mm_set1_epi32(0);
    const __m128i vMaxX = _mm_set1_epi32(grid.gridDimX - 1);
    const __m128i vMaxY = _mm_set1_epi32(grid.gridDimY - 1);
    const __m128i vMaxZ = _mm_set1_epi32(grid.gridDimZ - 1);

    const __m128i vDimX = _mm_set1_epi32(grid.gridDimX);
    const __m128i vDimXY = _mm_set1_epi32(grid.gridDimX * grid.gridDimY);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      __m128 px = _mm_loadu_ps(&ps.posX[i]);
      __m128 py = _mm_loadu_ps(&ps.posY[i]);
      __m128 pz = _mm_loadu_ps(&ps.posZ[i]);

      // (pos - origin) / cellSize
      px = _mm_mul_ps(_mm_sub_ps(px, vOriginX), vInvCell);
      py = _mm_mul_ps(_mm_sub_ps(py, vOriginY), vInvCell);
      pz = _mm_mul_ps(_mm_sub_ps(pz, vOriginZ), vInvCell);

      // floor
      __m128i cx = _mm_cvttps_epi32(px);
      __m128i cy = _mm_cvttps_epi32(py);
      __m128i cz = _mm_cvttps_epi32(pz);

      // clamp
      cx = _mm_min_epi32(_mm_max_epi32(cx, vZero), vMaxX);
      cy = _mm_min_epi32(_mm_max_epi32(cy, vZero), vMaxY);
      cz = _mm_min_epi32(_mm_max_epi32(cz, vZero), vMaxZ);

      // hash = cz * dimXY + cy * dimX + cx
      __m128i hash = _mm_add_epi32(
          _mm_add_epi32(
              _mm_mullo_epi32(cz, vDimXY),
              _mm_mullo_epi32(cy, vDimX)),
          cx);

      _mm_storeu_si128((__m128i*)&grid.cellHash[i], hash);

      grid.particleIndex[i+0] = i+0;
      grid.particleIndex[i+1] = i+1;
      grid.particleIndex[i+2] = i+2;
      grid.particleIndex[i+3] = i+3;
    }

    int scalarStart = N4;
  #elif CPU_SIMD_NEON
    // --- NEON path: 4 particles per iteration ---------------------------------
    const float32x4_t vOriginX = vdupq_n_f32(grid.originX);
    const float32x4_t vOriginY = vdupq_n_f32(grid.originY);
    const float32x4_t vOriginZ = vdupq_n_f32(grid.originZ);
    const float32x4_t vInvCell = vdupq_n_f32(1.0f / grid.cellSize);

    const int32x4_t vZero = vdupq_n_s32(0);
    const int32x4_t vMaxX = vdupq_n_s32(grid.gridDimX - 1);
    const int32x4_t vMaxY = vdupq_n_s32(grid.gridDimY - 1);
    const int32x4_t vMaxZ = vdupq_n_s32(grid.gridDimZ - 1);

    const int32x4_t vDimX = vdupq_n_s32(grid.gridDimX);
    const int32x4_t vDimXY = vdupq_n_s32(grid.gridDimX * grid.gridDimY);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float32x4_t px = vld1q_f32(&ps.posX[i]);
      float32x4_t py = vld1q_f32(&ps.posY[i]);
      float32x4_t pz = vld1q_f32(&ps.posZ[i]);

      px = vmulq_f32(vsubq_f32(px, vOriginX), vInvCell);
      py = vmulq_f32(vsubq_f32(py, vOriginY), vInvCell);
      pz = vmulq_f32(vsubq_f32(pz, vOriginZ), vInvCell);

      int32x4_t cx = vcvtq_s32_f32(px);
      int32x4_t cy = vcvtq_s32_f32(py);
      int32x4_t cz = vcvtq_s32_f32(pz);

      cx = vminq_s32(vmaxq_s32(cx, vZero), vMaxX);
      cy = vminq_s32(vmaxq_s32(cy, vZero), vMaxY);
      cz = vminq_s32(vmaxq_s32(cz, vZero), vMaxZ);

      int32x4_t hash = vaddq_s32(
          vaddq_s32(vmulq_s32(cz, vDimXY), vmulq_s32(cy, vDimX)),
          cx);

      vst1q_s32((int32_t*)&grid.cellHash[i], hash);

      grid.particleIndex[i+0] = i+0;
      grid.particleIndex[i+1] = i+1;
      grid.particleIndex[i+2] = i+2;
      grid.particleIndex[i+3] = i+3;
    }

    int scalarStart = N4;
  #else
    int scalarStart = 0;
  #endif

  // Scalar remainder (or full range when no SIMD)
  buildGrid_hashScalar(grid, ps, scalarStart, shellN);

  // Sort + sortedHash copy + cell ranges (identical in SISD and SIMD)
  buildGrid_sortAndRanges(grid, shellN);

  prevN = shellN;  // remember for sparse clear on next call
}

// ============================================================================
// neighborCollision - moved to shared/simulation_CPU.cpp
// ============================================================================

// ============================================================================
// integrateGround - FUSED: semi-implicit Euler + ground-plane collision
// ============================================================================
/**
 * @brief Fused integration + ground collision: Euler step, then ground contact check.
 * @param ps      ParticleSystem (reads force/mass, writes pos/vel).
 * @param shellN  Number of shell particles.
 * @param params  SimParams.
 */
void integrateGround(ParticleSystem &ps, int shellN, const SimParams &params) {
  float dt = params.dt;
  float e  = params.restitution;
  float mu = params.friction;

  float sn = params.slopeSin;
  float cs = params.slopeCos;
  float H  = params.slopeHeight;

  #if CPU_SIMD_X86
    // --- SSE path: 4 particles per iteration ----------------------------------
    const __m128 vDt = _mm_set1_ps(dt);
    const __m128 vSn = _mm_set1_ps(sn);
    const __m128 vCs = _mm_set1_ps(cs);
    const __m128 vH  = _mm_set1_ps(H);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {

      __m128 px = _mm_loadu_ps(&ps.posX[i]);
      __m128 py = _mm_loadu_ps(&ps.posY[i]);
      __m128 pz = _mm_loadu_ps(&ps.posZ[i]);

      __m128 vx = _mm_loadu_ps(&ps.velX[i]);
      __m128 vy = _mm_loadu_ps(&ps.velY[i]);
      __m128 vz = _mm_loadu_ps(&ps.velZ[i]);

      __m128 fx = _mm_loadu_ps(&ps.forceX[i]);
      __m128 fy = _mm_loadu_ps(&ps.forceY[i]);
      __m128 fz = _mm_loadu_ps(&ps.forceZ[i]);

      __m128 m  = _mm_loadu_ps(&ps.mass[i]);
      __m128 inv_m = _mm_div_ps(_mm_set1_ps(1.0f), m);

      // integrate velocity
      vx = _mm_add_ps(vx, _mm_mul_ps(_mm_mul_ps(fx, inv_m), vDt));
      vy = _mm_add_ps(vy, _mm_mul_ps(_mm_mul_ps(fy, inv_m), vDt));
      vz = _mm_add_ps(vz, _mm_mul_ps(_mm_mul_ps(fz, inv_m), vDt));

      // integrate position
      px = _mm_add_ps(px, _mm_mul_ps(vx, vDt));
      py = _mm_add_ps(py, _mm_mul_ps(vy, vDt));
      pz = _mm_add_ps(pz, _mm_mul_ps(vz, vDt));

      // distance from plane
      __m128 dist = _mm_sub_ps(
          _mm_add_ps(_mm_mul_ps(px, vSn), _mm_mul_ps(py, vCs)),
          vH);

      __m128 r = _mm_loadu_ps(&ps.radius[i]);

      __m128 mask = _mm_cmplt_ps(dist, r);
      int msk = _mm_movemask_ps(mask);

      if (msk != 0) {
        for (int k = 0; k < 4; k++) {
          int idx = i + k;

          if (!(msk & (1 << k))) continue;
          if (ps.state[idx] != ACTIVE) continue;

          float vx_s = vx.m128_f32[k];
          float vy_s = vy.m128_f32[k];
          float vz_s = vz.m128_f32[k];

          float px_s = px.m128_f32[k];
          float py_s = py.m128_f32[k];
          float pz_s = pz.m128_f32[k];

          float dist_s = px_s * sn + py_s * cs - H;
          float r_s = ps.radius[idx];

          if (dist_s < r_s) {
            float pen = r_s - dist_s;
            px_s += pen * sn;
            py_s += pen * cs;

            float vn = vx_s * sn + vy_s * cs;

            if (vn < 0.0f) {
              float dvn = -(1.0f + e) * vn;

              float vtX = vx_s - vn * sn;
              float vtY = vy_s - vn * cs;
              float vtZ = vz_s;

              float vtMag = sqrtf(vtX*vtX + vtY*vtY + vtZ*vtZ);
              float fricImp = mu * fabsf(dvn);

              if (vtMag > 1e-8f && fricImp < vtMag) {
                float scale = 1.0f - fricImp / vtMag;
                vtX *= scale; vtY *= scale; vtZ *= scale;
              } else if (vtMag > 1e-8f) {
                vtX = vtY = vtZ = 0.0f;
              }

              vx_s = vtX + (vn + dvn) * sn;
              vy_s = vtY + (vn + dvn) * cs;
              vz_s = vtZ;
            }
          }

          ps.posX[idx] = px_s;
          ps.posY[idx] = py_s;
          ps.posZ[idx] = pz_s;

          ps.velX[idx] = vx_s;
          ps.velY[idx] = vy_s;
          ps.velZ[idx] = vz_s;
        }
      } else {
        _mm_storeu_ps(&ps.posX[i], px);
        _mm_storeu_ps(&ps.posY[i], py);
        _mm_storeu_ps(&ps.posZ[i], pz);

        _mm_storeu_ps(&ps.velX[i], vx);
        _mm_storeu_ps(&ps.velY[i], vy);
        _mm_storeu_ps(&ps.velZ[i], vz);
      }
    }

    int scalarStart = N4;
  #elif CPU_SIMD_NEON
    // --- NEON path: 4 particles per iteration ---------------------------------
    const float32x4_t vDt = vdupq_n_f32(dt);
    const float32x4_t vSn = vdupq_n_f32(sn);
    const float32x4_t vCs = vdupq_n_f32(cs);
    const float32x4_t vH  = vdupq_n_f32(H);

    const int N4 = shellN & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float32x4_t px = vld1q_f32(&ps.posX[i]);
      float32x4_t py = vld1q_f32(&ps.posY[i]);
      float32x4_t pz = vld1q_f32(&ps.posZ[i]);

      float32x4_t vx = vld1q_f32(&ps.velX[i]);
      float32x4_t vy = vld1q_f32(&ps.velY[i]);
      float32x4_t vz = vld1q_f32(&ps.velZ[i]);

      float32x4_t fx = vld1q_f32(&ps.forceX[i]);
      float32x4_t fy = vld1q_f32(&ps.forceY[i]);
      float32x4_t fz = vld1q_f32(&ps.forceZ[i]);

      float32x4_t m  = vld1q_f32(&ps.mass[i]);
      float32x4_t inv_m = vdivq_f32(vdupq_n_f32(1.0f), m);

      vx = vaddq_f32(vx, vmulq_f32(vmulq_f32(fx, inv_m), vDt));
      vy = vaddq_f32(vy, vmulq_f32(vmulq_f32(fy, inv_m), vDt));
      vz = vaddq_f32(vz, vmulq_f32(vmulq_f32(fz, inv_m), vDt));

      px = vaddq_f32(px, vmulq_f32(vx, vDt));
      py = vaddq_f32(py, vmulq_f32(vy, vDt));
      pz = vaddq_f32(pz, vmulq_f32(vz, vDt));

      float32x4_t dist = vsubq_f32(
          vaddq_f32(vmulq_f32(px, vSn), vmulq_f32(py, vCs)),
          vH);

      float32x4_t r = vld1q_f32(&ps.radius[i]);

      uint32x4_t mask = vcltq_f32(dist, r);

      uint32_t msk =
          (vgetq_lane_u32(mask,0)?1:0) |
          (vgetq_lane_u32(mask,1)?2:0) |
          (vgetq_lane_u32(mask,2)?4:0) |
          (vgetq_lane_u32(mask,3)?8:0);

      if (msk != 0) {
        for (int k = 0; k < 4; k++) {
          int idx = i + k;
          if (!(msk & (1<<k))) continue;
          if (ps.state[idx] != ACTIVE) continue;

          // vgetq_lane_f32 requires a compile-time constant lane index on NEON
          float vx_s, vy_s, vz_s, px_s, py_s, pz_s;
          switch (k) {
            case 0: vx_s=vgetq_lane_f32(vx,0); vy_s=vgetq_lane_f32(vy,0); vz_s=vgetq_lane_f32(vz,0);
                    px_s=vgetq_lane_f32(px,0); py_s=vgetq_lane_f32(py,0); pz_s=vgetq_lane_f32(pz,0); break;
            case 1: vx_s=vgetq_lane_f32(vx,1); vy_s=vgetq_lane_f32(vy,1); vz_s=vgetq_lane_f32(vz,1);
                    px_s=vgetq_lane_f32(px,1); py_s=vgetq_lane_f32(py,1); pz_s=vgetq_lane_f32(pz,1); break;
            case 2: vx_s=vgetq_lane_f32(vx,2); vy_s=vgetq_lane_f32(vy,2); vz_s=vgetq_lane_f32(vz,2);
                    px_s=vgetq_lane_f32(px,2); py_s=vgetq_lane_f32(py,2); pz_s=vgetq_lane_f32(pz,2); break;
            default:vx_s=vgetq_lane_f32(vx,3); vy_s=vgetq_lane_f32(vy,3); vz_s=vgetq_lane_f32(vz,3);
                    px_s=vgetq_lane_f32(px,3); py_s=vgetq_lane_f32(py,3); pz_s=vgetq_lane_f32(pz,3); break;
          }

          float dist_s = px_s * sn + py_s * cs - H;
          float r_s = ps.radius[idx];

          if (dist_s < r_s) {
            float pen = r_s - dist_s;
            px_s += pen * sn;
            py_s += pen * cs;

            float vn = vx_s * sn + vy_s * cs;

            if (vn < 0.0f) {
              float dvn = -(1.0f + e) * vn;

              float vtX = vx_s - vn * sn;
              float vtY = vy_s - vn * cs;
              float vtZ = vz_s;

              float vtMag = sqrtf(vtX*vtX + vtY*vtY + vtZ*vtZ);
              float fricImp = mu * fabsf(dvn);

              if (vtMag > 1e-8f && fricImp < vtMag) {
                float scale = 1.0f - fricImp / vtMag;
                vtX *= scale; vtY *= scale; vtZ *= scale;
              } else if (vtMag > 1e-8f) {
                vtX = vtY = vtZ = 0.0f;
              }

              vx_s = vtX + (vn + dvn) * sn;
              vy_s = vtY + (vn + dvn) * cs;
              vz_s = vtZ;
            }
          }

          ps.posX[idx] = px_s;
          ps.posY[idx] = py_s;
          ps.posZ[idx] = pz_s;

          ps.velX[idx] = vx_s;
          ps.velY[idx] = vy_s;
          ps.velZ[idx] = vz_s;
        }
      } else {
        vst1q_f32(&ps.posX[i], px);
        vst1q_f32(&ps.posY[i], py);
        vst1q_f32(&ps.posZ[i], pz);

        vst1q_f32(&ps.velX[i], vx);
        vst1q_f32(&ps.velY[i], vy);
        vst1q_f32(&ps.velZ[i], vz);
      }
    }

    int scalarStart = N4;
  #else
    int scalarStart = 0;
  #endif

  integrateGround_scalar(ps, scalarStart, shellN, params);
}

// ============================================================================
// captureFromSnowpack - probabilistic capture of particles into the shell
// ============================================================================
/**
 * @brief Probabilistic capture of snowpack particles into the shell.
 * @param sp                 Snowpack (marks captured particles as dead).
 * @param shell              Shell ParticleSystem to append new particles to.
 * @param shellN             Current number of shell particles.
 * @param ball               Current snowball state.
 * @param params             SimParams providing sticking model coefficients.
 * @param outCapturedMass    Output: total mass of captured particles.
 * @param frame              Current frame number (used for hash-based PRNG).
 * @param spOffset           Offset into snowpack arrays to start processing (for splitting into multiple launches).
 * @param spCount            Number of snowpack particles to process in this launch (for splitting into multiple launches).
 * @return Number of particles captured this frame.
 */
int captureFromSnowpack(
    Snowpack &sp,
    ParticleSystem &shell, int shellN,
    const SnowballState &ball,
    const SimParams &params,
    float *outCapturedMass,
    int frame,
    int spOffset, int spCount)
{
  int captureCount = 0;
  float captureMass = 0.0f;

  float ballRadius = ball.radius;

  // For small sp.count it's more convenient using only the last scalar function

  #if CPU_SIMD_X86
    // --- SSE path: 4 particles per iteration ----------------------------------
    const __m128 vBpx = _mm_set1_ps(ball.posX);
    const __m128 vBpy = _mm_set1_ps(ball.posY);
    const __m128 vBpz = _mm_set1_ps(ball.posZ);
    const __m128 vBallR = _mm_set1_ps(ballRadius);
    const __m128 vMargin = _mm_set1_ps(0.3f);

    const int N4 = sp.count & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static) reduction(+:captureCount, captureMass)
    #endif
    for (int i = 0; i < N4; i += 4) {
      __m128 px = _mm_loadu_ps(&sp.posX[i]);
      __m128 py = _mm_loadu_ps(&sp.posY[i]);
      __m128 pz = _mm_loadu_ps(&sp.posZ[i]);
      __m128 spr = _mm_loadu_ps(&sp.radius[i]);

      __m128 dx = _mm_sub_ps(px, vBpx);
      __m128 dy = _mm_sub_ps(py, vBpy);
      __m128 dz = _mm_sub_ps(pz, vBpz);

      __m128 dist2 = _mm_add_ps(
          _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
          _mm_mul_ps(dz, dz));

      __m128 captureR = _mm_add_ps(_mm_add_ps(vBallR, spr), vMargin);
      __m128 captureR2 = _mm_mul_ps(captureR, captureR);
      __m128 mask = _mm_cmple_ps(dist2, captureR2);
      int m = _mm_movemask_ps(mask);

      if (m == 0) continue;

      // fallback scalar per lane
      for (int k = 0; k < 4; k++) {
        int idx = i + k;

        if (!(m & (1 << k))) continue;
        if (sp.alive[idx] != AVAILABLE) continue;

        float pxs = sp.posX[idx];
        float pys = sp.posY[idx];
        float pzs = sp.posZ[idx];
        float sprs = sp.radius[idx];

        float dxs = pxs - ball.posX;
        float dys = pys - ball.posY;
        float dzs = pzs - ball.posZ;

        float dist2s = dxs*dxs + dys*dys + dzs*dzs;

        float captureR = ballRadius + sprs + 0.3f;
        if (dist2s > captureR * captureR) continue;

        float dist = sqrtf(dist2s);
        if (dist < 1e-8f) continue;

        float vrelMag = sqrtf(ball.velX*ball.velX + ball.velY*ball.velY + ball.velZ*ball.velZ);

        float w = sp.wetness[idx];
        float logit = params.stickK0
                    + params.stickK1 * w
                    - params.stickK2 * vrelMag
                    + params.stickRadiusBoost * ballRadius;

        float clamp = params.logitClamp;
        logit = fminf(fmaxf(logit, -clamp), clamp);

        float P_stick = 1.0f / (1.0f + expf(-logit));

        uint32_t h = (uint32_t)idx ^ (uint32_t)(frame * 2654435761u);
        h ^= h >> 16; h *= 0x45d9f3bu;
        h ^= h >> 16; h *= 0x45d9f3bu;
        h ^= h >> 16;
        float rnd = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

        if (rnd >= P_stick) continue;

        int slot = captureCount;
        if (params.maxCapturePerFrm > 0 && slot >= params.maxCapturePerFrm) break;
        if (shellN + slot >= shell.capacity) break;

        sp.alive[idx] = CONSUMED;
        captureMass += sp.mass[idx];

        int si = shellN + slot;

        float invDist = 1.0f / dist;
        float nx = dxs * invDist;
        float ny = dys * invDist;
        float nz = dzs * invDist;

        float surfDist = ballRadius + sprs;

        shell.posX[si] = ball.posX + nx * surfDist;
        shell.posY[si] = ball.posY + ny * surfDist;
        shell.posZ[si] = ball.posZ + nz * surfDist;

        float rx = nx * surfDist;
        float ry = ny * surfDist;
        float rz = nz * surfDist;

        shell.velX[si] = ball.velX + (ball.omegaY * rz - ball.omegaZ * ry);
        shell.velY[si] = ball.velY + (ball.omegaZ * rx - ball.omegaX * rz);
        shell.velZ[si] = ball.velZ + (ball.omegaX * ry - ball.omegaY * rx);

        shell.forceX[si] = 0.0f;
        shell.forceY[si] = 0.0f;
        shell.forceZ[si] = 0.0f;

        shell.mass[si]    = sp.mass[idx];
        shell.radius[si]  = sprs;
        shell.state[si]   = ACTIVE;
        shell.wetness[si] = w;

        quatInvRotateVec(ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
                         nx, ny, nz,
                         shell.attachLocalX[si], shell.attachLocalY[si], shell.attachLocalZ[si]);

        captureCount++;
      }
    }

    int scalarStart = N4;
  #elif CPU_SIMD_NEON
    // --- NEON path: 4 particles per iteration ---------------------------------
    const float32x4_t vBpx = vdupq_n_f32(ball.posX);
    const float32x4_t vBpy = vdupq_n_f32(ball.posY);
    const float32x4_t vBpz = vdupq_n_f32(ball.posZ);
    const float32x4_t vBallR = vdupq_n_f32(ball.radius);
    const float32x4_t vMargin = vdupq_n_f32(0.3f);

    const int N4 = sp.count & ~3;

    #ifdef HAS_OPENMP
      #pragma omp parallel for schedule(static) reduction(+:captureCount, captureMass)
    #endif
    for (int i = 0; i < N4; i += 4) {
      float32x4_t px = vld1q_f32(&sp.posX[i]);
      float32x4_t py = vld1q_f32(&sp.posY[i]);
      float32x4_t pz = vld1q_f32(&sp.posZ[i]);
      float32x4_t spr = vld1q_f32(&sp.radius[i]);

      float32x4_t dx = vsubq_f32(px, vBpx);
      float32x4_t dy = vsubq_f32(py, vBpy);
      float32x4_t dz = vsubq_f32(pz, vBpz);

      float32x4_t dist2 = vaddq_f32(
          vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy)),
          vmulq_f32(dz, dz));

      float32x4_t captureR = vaddq_f32(vaddq_f32(vBallR, spr), vMargin);
      float32x4_t captureR2 = vmulq_f32(captureR, captureR);
      uint32x4_t mask = vcleq_f32(dist2, captureR2);

      uint32_t m =
          (vgetq_lane_u32(mask, 0) ? 1 : 0) |
          (vgetq_lane_u32(mask, 1) ? 2 : 0) |
          (vgetq_lane_u32(mask, 2) ? 4 : 0) |
          (vgetq_lane_u32(mask, 3) ? 8 : 0);

      if (m == 0) continue;

      for (int k = 0; k < 4; k++) {
        int idx = i + k;

        if (!(m & (1 << k))) continue;
        if (sp.alive[idx] != AVAILABLE) continue;

        float pxs = sp.posX[idx];
        float pys = sp.posY[idx];
        float pzs = sp.posZ[idx];
        float sprs = sp.radius[idx];

        float dxs = pxs - ball.posX;
        float dys = pys - ball.posY;
        float dzs = pzs - ball.posZ;

        float dist2s = dxs*dxs + dys*dys + dzs*dzs;

        float captureR = ball.radius + sprs + 0.3f;
        if (dist2s > captureR * captureR) continue;

        float dist = sqrtf(dist2s);
        if (dist < 1e-8f) continue;

        float vrelMag = sqrtf(ball.velX*ball.velX + ball.velY*ball.velY + ball.velZ*ball.velZ);

        float w = sp.wetness[idx];
        float logit = params.stickK0
                    + params.stickK1 * w
                    - params.stickK2 * vrelMag
                    + params.stickRadiusBoost * ball.radius;

        float clamp = params.logitClamp;
        logit = fminf(fmaxf(logit, -clamp), clamp);

        float P_stick = 1.0f / (1.0f + expf(-logit));

        uint32_t h = (uint32_t)idx ^ (uint32_t)(frame * 2654435761u);
        h ^= h >> 16; h *= 0x45d9f3bu;
        h ^= h >> 16; h *= 0x45d9f3bu;
        h ^= h >> 16;
        float rnd = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;

        if (rnd >= P_stick) continue;

        int slot = captureCount;
        if (params.maxCapturePerFrm > 0 && slot >= params.maxCapturePerFrm) break;
        if (shellN + slot >= shell.capacity) break;

        sp.alive[idx] = CONSUMED;
        captureMass += sp.mass[idx];

        int si = shellN + slot;

        float invDist = 1.0f / dist;
        float nx = dxs * invDist;
        float ny = dys * invDist;
        float nz = dzs * invDist;

        float surfDist = ball.radius + sprs;

        shell.posX[si] = ball.posX + nx * surfDist;
        shell.posY[si] = ball.posY + ny * surfDist;
        shell.posZ[si] = ball.posZ + nz * surfDist;

        float rx = nx * surfDist;
        float ry = ny * surfDist;
        float rz = nz * surfDist;

        shell.velX[si] = ball.velX + (ball.omegaY * rz - ball.omegaZ * ry);
        shell.velY[si] = ball.velY + (ball.omegaZ * rx - ball.omegaX * rz);
        shell.velZ[si] = ball.velZ + (ball.omegaX * ry - ball.omegaY * rx);

        shell.forceX[si] = 0.0f;
        shell.forceY[si] = 0.0f;
        shell.forceZ[si] = 0.0f;

        shell.mass[si]    = sp.mass[idx];
        shell.radius[si]  = sprs;
        shell.state[si]   = ACTIVE;
        shell.wetness[si] = w;

        quatInvRotateVec(ball.quatW, ball.quatX, ball.quatY, ball.quatZ,
                         nx, ny, nz,
                         shell.attachLocalX[si], shell.attachLocalY[si], shell.attachLocalZ[si]);

        captureCount++;
      }
    }

    int scalarStart = N4;
  #else
    int scalarStart = 0;
  #endif

  // Scalar remainder (or full range without SIMD) + SISD-identical logic
  captureFromSnowpack_scalar(sp, shell, shellN, ball, frame, spOffset, spOffset + spCount, captureCount, captureMass, params);

  *outCapturedMass = captureMass;
  return captureCount;
}