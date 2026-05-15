#ifndef SIMULATION_CPU_H
  #define SIMULATION_CPU_H

  void allocateParticleSystem(ParticleSystem& ps, int capacity);
  void freeParticleSystem(ParticleSystem& ps);
  void allocateSnowpack(Snowpack& sp, int count);
  void freeSnowpack(Snowpack& sp);
  void allocateGrid(GridData& grid, const SimParams& params, int shellCapacity);
  void freeGrid(GridData& grid);
  void growParticleSystem(ParticleSystem& ps, int activeCount, int newCapacity);
  void growGrid(GridData& grid, int newCapacity);

  void quatRotateVec(
      float qw, float qx, float qy, float qz,
      float vx, float vy, float vz,
      float &ox, float &oy, float &oz);
  void quatInvRotateVec(
      float qw, float qx, float qy, float qz,
      float vx, float vy, float vz,
      float &ox, float &oy, float &oz);
  
  float lcgRandom(uint32_t &state);

  void initSnowpack(Snowpack& sp, const SimParams& params);
  void sortSnowpackByPosX(Snowpack& sp);
  void forcesTether(ParticleSystem& shell, const SnowballState& ball, int shellN, const SimParams& params);
  void buildGrid(GridData& grid, const ParticleSystem& ps, int shellN);
  void neighborCollision(ParticleSystem& ps, const GridData& grid, int shellN, const SimParams& params);
  void integrateGround(ParticleSystem& ps, int shellN, const SimParams& params);
  int captureFromSnowpack(
      Snowpack& sp, 
      ParticleSystem& shell, int shellN,
      const SnowballState& ball,
      const SimParams& params,
      float* outCapturedMass,
      int frame,
      int spOffset, int spCount);

  // --- Scalar kernel helpers (shared between SISD and SIMD remainder) ------
  void initSnowpack_scalar(Snowpack& sp, int start, int end, const SimParams& params);
  void forcesTether_scalar(ParticleSystem& shell, const SnowballState& ball, int start, int end, const SimParams& params);
  void integrateGround_scalar(ParticleSystem& ps, int start, int end, const SimParams& params);
  void buildGrid_hashScalar(GridData& grid, const ParticleSystem& ps, int start, int end);
  void buildGrid_sortAndRanges(GridData& grid, int shellN);
  void buildGrid_scalar(GridData& grid, const ParticleSystem& ps, int shellN);
  void captureFromSnowpack_scalar(
      Snowpack& sp,
      ParticleSystem& shell, int shellN,
      const SnowballState& ball,
      int frame,
      int start, int end,
      int& captureCount, float& captureMass,
      const SimParams& params);

  int runSimulation(SimParams& params);
#endif // SIMULATION_CPU_H