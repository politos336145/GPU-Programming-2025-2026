#ifndef GRID_CUH
  #define GRID_CUH

  #include "../../../shared/include/types.h"

  void allocateGrid(GridData& grid, int shellCapacity, const SimParams& params);
  void freeGrid(GridData& grid);
  void resizeGrid(GridData& grid, int newShellCapacity, const SimParams& params);

  void launchBuildGrid(GridData& grid, const ParticleSystem& ps, int shellN, cudaStream_t stream = 0);
  
  void printGridStats(const GridData& grid, int shellN, int frame);
#endif // GRID_CUH
