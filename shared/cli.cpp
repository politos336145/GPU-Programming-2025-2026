// ============================================================================
// CLI argument helpers
// Provides simple functions to parse integer, float, and boolean arguments from the command line.
// ============================================================================

#include "include/cli.h"

#include <cstring>
#include <cstdlib>

/**
 * @brief Parses an integer argument from the command line.
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * @param flag  The flag to search for (e.g. "--particles").
 * @param def   Default value returned if the flag is not found.
 * @return The parsed integer value, or def if the flag is absent.
 */
int argInt(int argc, char** argv, const char* flag, int def) {
  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], flag) == 0)
      return atoi(argv[i + 1]);
  }
  
  return def;
}

/**
 * @brief Parses a float argument from the command line.
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * @param flag  The flag to search for (e.g. "--dt").
 * @param def   Default value returned if the flag is not found.
 * @return The parsed float value, or def if the flag is absent.
 */
float argFloat(int argc, char** argv, const char* flag, float def) {
  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], flag) == 0)
      return (float)atof(argv[i + 1]);
  }
  
  return def;
}

/**
 * @brief Parses a boolean flag from the command line.
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * @param flag  The flag to search for (e.g. "--no-log").
 * @return true if the flag is present, false otherwise.
 */
bool argBool(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], flag) == 0)
      return true;
  }
  
  return false;
}