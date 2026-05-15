#ifndef KERNEL_CUH
  #define KERNEL_CUH

  #include "../../../shared/include/types.h"

  // ============================================================================
  // Snowpack Kernels (from snowpack.cu)
  // ============================================================================

  __global__ void fillIotaKernel(uint32_t *arr, int N);
  
  __global__ void gatherFloatKernel(
      float *__restrict__ dst, const float *__restrict__ src,
      const uint32_t *__restrict__ indices,
      int N);
  
  __global__ void gatherIntKernel(
      int *__restrict__ dst, const int *__restrict__ src,
      const uint32_t *__restrict__ indices,
      int N);
  
  __global__ void initSnowpackKernel(
      float *__restrict__ posX, float *__restrict__ posY, float *__restrict__ posZ,
      float *__restrict__ radius,
      float *__restrict__ mass,
      float *__restrict__ wetness,
      SnowpackParticleState *__restrict__ alive,
      unsigned int seed);
  
  __global__ void captureFromSnowpackKernel(
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
      int spCount);

  // ============================================================================
  // Grid Kernels (from grid.cu)
  // ============================================================================

  __global__ void computeHashKernel(
      uint32_t *__restrict__ cellHash,
      uint32_t *__restrict__ particleIndex,
      const float *__restrict__ posX, const float *__restrict__ posY, const float *__restrict__ posZ,
      float originX, float originY, float originZ,
      float cellSize,
      int gridDimX, int gridDimY, int gridDimZ,
      int hashTableMask,
      int shellN);
  
  __global__ void clearPrevCellsKernel(
      const uint32_t *__restrict__ prevSortedHash,
      int *__restrict__ cellStart, int *__restrict__ cellEnd,
      int shellN);

  __global__ void buildCellRangesKernel(
      const uint32_t *__restrict__ cellHash,
      int *__restrict__ cellStart, int *__restrict__ cellEnd,
      int shellN);

  // ============================================================================
  // Simulation Kernels (from simulation.cu)
  // ============================================================================

  __global__ void neighborCollisionKernel(
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
      int shellN);

  __global__ void bruteForceCollisionKernel(
      const float *__restrict__ posX, const float *__restrict__ posY, const float *__restrict__ posZ,
      const float *__restrict__ velX, const float *__restrict__ velY, const float *__restrict__ velZ,
      float *__restrict__ forceX, float *__restrict__ forceY, float *__restrict__ forceZ,
      const float *__restrict__ radius,
      const ShellParticleState *__restrict__ state,
      int shellN);

  __global__ void shellPhysicsFusedKernel(
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
      int shellN);

  __global__ void forceTetherKernel(
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
      int shellN);

  __global__ void integrateGroundKernel(
      float *__restrict__ posX,  float *__restrict__ posY,  float *__restrict__ posZ,
      float *__restrict__ velX,  float *__restrict__ velY,  float *__restrict__ velZ,
      const float *__restrict__ forceX, const float *__restrict__ forceY, const float *__restrict__ forceZ,
      const float *__restrict__ mass,
      const float *__restrict__ radius,
      const ShellParticleState *__restrict__ state,
      int shellN);
#endif // KERNEL_CUH
