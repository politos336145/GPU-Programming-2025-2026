#ifndef SIMULATION_CUH
  #define SIMULATION_CUH

  #include "../../../shared/include/types.h"

  void copyParamsToGPU(const SimParams& params);

  void allocateParticleSystem(ParticleSystem& ps, int capacity);
  void freeParticleSystem(ParticleSystem& ps);
  void growShell(ParticleSystem& shell, int shellN, int newCapacity);

  void launchNeighborCollision(ParticleSystem& ps, const GridData& grid, int shellN, cudaStream_t stream = 0);
  void launchBruteForceCollision(ParticleSystem& ps, int shellN, cudaStream_t stream = 0);
  void launchShellPhysicsFused(ParticleSystem& ps, const SnowballState& ball, float ballRadius, int shellN, cudaStream_t stream = 0);
  void launchForcesTether(ParticleSystem& shell, const SnowballState& ball, int shellN, cudaStream_t stream = 0);
  void launchIntegrateGround(ParticleSystem& ps, int shellN, cudaStream_t stream = 0);
#endif // SIMULATION_CUH
