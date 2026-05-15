#include "../../shared/include/global.h"
#include "../../shared/include/simulation_CPU.h"

// ============================================================================
// initSnowpack - place particles on slope surface with jitter and wetness
// ============================================================================
/**
 * @brief Initialize snowpack particles on the slope surface.
 * @param sp      Snowpack to populate.
 * @param params  SimParams providing slope geometry and particle properties.
 */
void initSnowpack(Snowpack &sp, const SimParams &params) {
  initSnowpack_scalar(sp, 0, sp.count, params);
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
  forcesTether_scalar(shell, ball, 0, shellN, params);
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
  buildGrid_scalar(grid, ps, shellN);
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
  integrateGround_scalar(ps, 0, shellN, params);
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
  int   captureCount = 0;
  float captureMass  = 0.0f;

  captureFromSnowpack_scalar(sp, shell, shellN, ball, frame, spOffset, spOffset + spCount, captureCount, captureMass, params);

  *outCapturedMass = captureMass;
  return captureCount;
}