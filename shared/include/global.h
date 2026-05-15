#ifndef GLOBAL_H
  #define GLOBAL_H

  #include "types.h"
  
  #include <string>

  SimParams defaultParams(void);
  std::string newTimestamp(void);
  
  void initSnowball(SnowballState& ball, const SimParams& params);
  void updateSnowball(SnowballState& ball, const SimParams& params);

  void computeGridDimensions(GridData& grid, const SimParams& params);

  void updateCoreMassAfterCapture(SnowballState& ball, int capturedCount, float capturedMass, float snowDensity);

  void printSimulationSummary(const SnowballState& ball, int shellN);
  void setParamsFromCLI(SimParams* params, int argc, char** argv);
  void printHelp(void);
  void printConfig(const SimParams& params);
#endif // GLOBAL_H