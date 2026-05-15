// ============================================================================
// Logging implementation
// This file implements the Logger class, which provides functionality to log
// simulation stats to both stdout and an optional CSV file for later analysis.
// ============================================================================

#include "include/logging.h"

/**
 * @brief Construct a logger with no CSV file open.
 */
Logger::Logger(void) : _csvFile(nullptr) {}

/**
 * @brief Destroy the logger and close any open CSV file.
 */
Logger::~Logger(void) { close(); }

/**
 * @brief Open a CSV file for logging simulation data.
 * @param filename  Path to the CSV file to create.
 * @return true if the file was opened successfully, false otherwise.
 */
bool Logger::openCSV(const char *filename) {
  _csvFile = fopen(filename, "w");
  if (!_csvFile) {
    fprintf(stderr, "Warning: could not open log file '%s'\n", filename);
    return false;
  }

  writeHeader();

  return true;
}

/**
 * @brief Write the CSV header row if a log file is open.
 */
void Logger::writeHeader(void) {
  const char *header =
    "frame,shell,snowpack_alive,ball_mass,ball_radius,"
    "ball_px,ball_py,ball_pz,"
    "ball_vx,ball_vy,ball_vz,"
    "frame_ms,"
    "capture_ms,forcesTether_ms,shellGrid_ms,"
    "shellCollision_ms,intGround_ms,coreUpdate_ms\n";

  if (_csvFile)
    fprintf(_csvFile, "%s", header);
}

/**
 * @brief Log one frame's stats to stdout and CSV (if open).
 * @param s  FrameStats struct with all per-frame simulation metrics.
 */
void Logger::logFrame(const FrameStats &s) {
  printf("[%5d] shell=%4d snowpack=%5d  ball: m=%.3f r=%.3f pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f)  %.2f ms"
         "  [Cap:%.3f F+T:%.3f Grid:%.3f Coll:%.3f I+G:%.3f Core:%.3f]\n",
    s.frame,
    s.shellCount,
    s.snowpackAlive,
    s.ballMass,
    s.ballRadius,
    s.ballPosX, s.ballPosY, s.ballPosZ,
    s.ballVelX, s.ballVelY, s.ballVelZ,
    s.frameTimeMs, s.captureMs, s.shellForcesTetherMs, s.shellGridMs, s.shellCollisionMs, s.shellIntGroundMs, s.coreUpdateMs);

  if (_csvFile) {
    fprintf(_csvFile, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
      s.frame,
      s.shellCount,
      s.snowpackAlive,
      s.ballMass,
      s.ballRadius,
      s.ballPosX, s.ballPosY, s.ballPosZ,
      s.ballVelX, s.ballVelY, s.ballVelZ,
      s.frameTimeMs, s.captureMs, s.shellForcesTetherMs, s.shellGridMs, s.shellCollisionMs, s.shellIntGroundMs, s.coreUpdateMs);
    
    fflush(_csvFile);
  }
}

/**
 * @brief Close the CSV file if it is open.
 */
void Logger::close(void) {
  if (_csvFile) {
    fclose(_csvFile);
    _csvFile = nullptr;
  }
}