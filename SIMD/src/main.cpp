#include "../../shared/include/global.h"
#include "../../shared/include/simulation_CPU.h"

#include <chrono>
#include <cstring> // for Linux
#include <string>

SimParams params = defaultParams();
std::string timestamp = newTimestamp();

// ============================================================================
// Main
// ============================================================================
/**
 * @brief Main entry point: parse CLI, run simulation loop.
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * @return 0 on success.
 */
int main(int argc, char** argv) {
  printf("======================================================================\n");
  printf("===      Angry Santa - Snowball and Avalanche Simulation           ===\n");
  printf("===    CPU SIMD: Optimization enabled (with OpenMP, with SIMD)     ===\n");
  printf("======================================================================\n");

  if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    printHelp();
    return 0;
  }
  
  setParamsFromCLI(&params, argc, argv);
  printConfig(params);

  auto progStart = std::chrono::high_resolution_clock::now();
  int ret = runSimulation(params);
  auto progEnd = std::chrono::high_resolution_clock::now();
  double progMs = std::chrono::duration<double, std::milli>(progEnd - progStart).count();
  printf("Total program time: %.3f s\n", progMs / 1000.0);
  
  return ret;
}