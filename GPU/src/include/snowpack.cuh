#ifndef SNOWPACK_CUH
  #define SNOWPACK_CUH

  #include "../../../shared/include/types.h"

  void allocateSnowpack(Snowpack& sp, int N);
  void freeSnowpack(Snowpack& sp);
  
  void sortSnowpackByPosX(Snowpack& sp);

  void launchInitSnowpack(Snowpack& sp, unsigned int seed);
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
      cudaStream_t stream = 0);
#endif // SNOWPACK_CUH
