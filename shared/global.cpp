#include "include/cli.h"
#include "include/global.h"
#include "include/types.h"

#include <cmath> // mandatory on LINUX
#include <ctime>

// ============================================================================
// defaultParams - returns a SimParams struct with reasonable default values
// ============================================================================
/**
 * @brief Build a SimParams struct with reasonable default values for all fields.
 * @return SimParams initialized with defaults (can be overridden by CLI).
 */
SimParams defaultParams(void) {
  SimParams p = {};
  p.gravity           = 9.81f;
  p.dt                = 0.0033f;
  p.damping           = 0.05f;
  p.stiffness         = 30000.0f;

  p.spawnLengthS     = 1000.0f;
  p.spawnWidthZ      = 10.0f;
  p.spawnStartS      = 0.0f;
  p.spawnThicknessN  = 0.0f;

  p.collisionDamping  = 50.0f;
  p.cohesion          = 72450.0f;
  p.cohesionRadius    = 0.06f;
  p.particleRadius    = 0.020f;
  p.particleMass      = 0.1f;
  p.restitution       = 0.15f;
  p.friction          = 0.14f;
  p.slopeAngleDeg     = 30.0f;
  p.slopeSin          = sin(p.slopeAngleDeg * M_PI / 180.0f);
  p.slopeCos          = cos(p.slopeAngleDeg * M_PI / 180.0f);
  p.slopeHeight       = 30.0f;  // plane passes at height 30 when x=0 - long slope
  p.cellSize          = 4 * p.particleRadius;
  p.stickK0           = 1.0f;   // base logit → ~73% at v=0, w=1
  p.stickK1           = 7.11f;  // wetness boost in logit
  p.stickK2           = 1.1f;   // velocity penalty (per m/s of |v_rel|)
  p.stickRadiusBoost  = 7.5f;   // logit boost per metre of ball radius
  p.particleFriction  = 0.04f;  // inter-particle Coulomb friction coefficient
  p.wetnessMin        = 0.0f;   // min initial wetness
  p.wetnessMax        = 1.0f;   // max initial wetness
  p.snowDensity       = 200.0f; // kg/m³ - snow density for ball radius
  p.traceInterval     = 0;      // disabled by default (enable via CLI)
  p.domainLength      = 150.0f; // max along-slope travel
  p.numParticles      = 1000000;
  p.logInterval       = 0;      // disabled by default (enable via CLI)

  p.shellTetherK     = 2717.0f;
  p.shellTetherDamp  = 50.0f;

  p.logitClamp       = 15.0f;
  p.maxCapturePerFrm = 150;

  p.rollingDrag  = 0.15f;
  p.aeroCoeff    = 0.3f;

  return p;
}

// ============================================================================
// newTimestamp - generate a timestamp string for log file naming
// ============================================================================
/**
 * @brief Generate a timestamp string in the format YYYYMMDD_HHMMSS for log file naming.
 * @return Timestamp string representing the current local time.
 */
std::string newTimestamp(void) {
  std::time_t now = std::time(nullptr);
  std::tm* tm_info = std::localtime(&now);
  char buffer[15];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", tm_info);

  return std::string(buffer);
}

// ============================================================================
// initSnowball - initialize the snowball rigid-body state on the slope surface
// ============================================================================
/**
 * @brief Initialize the snowball rigid-body state on the slope surface.
 * @param ball    SnowballState to initialize (position, velocity, mass, radius, quaternion).
 * @param params  SimParams providing slope geometry and snow density.
 */
void initSnowball(SnowballState &ball, const SimParams &params) {
  
  // Place snowball at s=1.0 along slope (near top, ahead of particle patch at s=2), on surface
  float sn = params.slopeSin;
  float cs = params.slopeCos;
  float H  = params.slopeHeight;
  float s0 = 1.0f; // start offset along slope (m)

  ball.radius = 0.03f;  // start very tiny - grows dramatically via avalanche
  // Derive initial mass from radius and snow density for consistency:
  // m = 4/3 * π * r³ * ρ
  ball.mass = (4.0f / 3.0f) * (float)M_PI * ball.radius * ball.radius * ball.radius * params.snowDensity;
  
  ball.posX = s0 * cs + ball.radius * sn;            // along slope + normal offset
  ball.posY = (H / cs) - s0 * sn + ball.radius * cs; // on surface + normal offset
  ball.posZ = 0.0f;
  
  ball.velX = 0.0f;
  ball.velY = 0.0f;
  ball.velZ = 0.0f;
  
  ball.capturedParticleCount = 0;

  // Identity quaternion (no rotation)
  ball.quatW = 1.0f;
  ball.quatX = 0.0f;
  ball.quatY = 0.0f;
  ball.quatZ = 0.0f;

  // Zero angular velocity
  ball.omegaX = 0.0f;
  ball.omegaY = 0.0f;
  ball.omegaZ = 0.0f;
}

// ============================================================================
// updateSnowball - move snowball under gravity along slope
// Includes rolling-without-slip angular velocity and quaternion integration.
// Slope normal N = (sinθ, cosθ, 0).  Downhill tangent T = (cosθ, -sinθ, 0).
// Plane equation: sinθ·x + cosθ·y = slopeHeight.
// ============================================================================
/**
 * @brief Update snowball rigid body under gravity along slope.
 *        Includes rolling-without-slip angular velocity and quaternion integration.
 * @param ball    SnowballState to update in place.
 * @param params  SimParams providing dt, gravity, slope geometry.
 */
void updateSnowball(SnowballState &ball, const SimParams &params) {
  float dt = params.dt;
  float g  = params.gravity;
  float sn = params.slopeSin;
  float cs = params.slopeCos;
  float H  = params.slopeHeight;

  // Acceleration along slope downhill: a = g·sinθ in direction T = (cosθ, -sinθ, 0)
  float aSlope = g * sn;
  ball.velX += aSlope * cs * dt;
  ball.velY += -aSlope * sn * dt;

  // ---- Rolling resistance (linear drag, proportional to velocity) ----
  // Models rolling friction + surface deformation losses.
  // v_terminal (this term alone) = g·sinθ / rollingDrag
  float rd = params.rollingDrag;
  ball.velX *= (1.0f - rd * dt);
  ball.velY *= (1.0f - rd * dt);
  ball.velZ *= (1.0f - rd * dt);

  // ---- Aerodynamic drag (quadratic, scales with frontal area R²) ----
  // F_aero = aeroCoeff * R² * v²  (opposes motion)
  // Δv_aero = -F_aero / m * dt = -aeroCoeff * R² * v² / m * dt
  // At large R this becomes the dominant braking term, giving physically
  // bounded terminal velocity even as the ball grows.
  if (params.aeroCoeff > 0.0f && ball.mass > 0.0f) {
    float vx = ball.velX;
    float vy = ball.velY;
    float vz = ball.velZ;
    float v2 = vx * vx + vy * vy + vz * vz;
    if (v2 > 1e-8f) {
      float v = sqrtf(v2);
      float R2 = ball.radius * ball.radius;
      float aeroDec = params.aeroCoeff * R2 * v / ball.mass * dt; // Δv magnitude
      
      // Clamp so drag never reverses the velocity in one step
      if (aeroDec > v) aeroDec = v;
      float scale = 1.0f - aeroDec / v;
      ball.velX *= scale;
      ball.velY *= scale;
      ball.velZ *= scale;
    }
  }

  ball.posX += ball.velX * dt;
  ball.posY += ball.velY * dt;
  ball.posZ += ball.velZ * dt;

  // Keep ball on slope surface: signed dist = sn·x + cs·y - H
  float dist = ball.posX * sn + ball.posY * cs - H;
  if (dist < ball.radius) {
    float pen = ball.radius - dist;
    ball.posX += pen * sn;
    ball.posY += pen * cs;

    // Cancel velocity into slope
    float vn = ball.velX * sn + ball.velY * cs;
    if (vn < 0.0f) {
      ball.velX -= vn * sn;
      ball.velY -= vn * cs;
    }
  }

  // Domain bound: stop at bottom of slope (Y <= 0)
  if (ball.posY < ball.radius) {
    ball.posY = ball.radius;
    if (ball.velY < 0.0f) ball.velY = 0.0f;
    ball.velX *= 0.95f; // friction at bottom
  }

  // ---- Rolling rotation: ω = (N_slope × v) / R ----
  // Slope normal N = (sinθ, cosθ, 0), v = (velX, velY, velZ)
  // cross(N_slope, v) = (cs*velZ, -sn*velZ, sn*velY - cs*velX)
  if (ball.radius > 1e-6f) {
    float invR = 1.0f / ball.radius;
    ball.omegaX = invR * (cs * ball.velZ);
    ball.omegaY = invR * (-sn * ball.velZ);
    ball.omegaZ = invR * (sn * ball.velY - cs * ball.velX);
  }

  // ---- Quaternion integration: q' = q + (dt/2) · (0, ω) ⊗ q ----
  float ox = ball.omegaX, oy = ball.omegaY, oz = ball.omegaZ;
  float qw = ball.quatW, qx = ball.quatX, qy = ball.quatY, qz = ball.quatZ;

  float halfDt = 0.5f * dt;
  float dqw = halfDt * (-ox * qx - oy * qy - oz * qz);
  float dqx = halfDt * ( ox * qw + oy * qz - oz * qy);
  float dqy = halfDt * (-ox * qz + oy * qw + oz * qx);
  float dqz = halfDt * ( ox * qy - oy * qx + oz * qw);

  ball.quatW = qw + dqw;
  ball.quatX = qx + dqx;
  ball.quatY = qy + dqy;
  ball.quatZ = qz + dqz;

  // Re-normalize quaternion to prevent numerical drift
  float qLen = sqrtf(ball.quatW * ball.quatW + ball.quatX * ball.quatX +
                     ball.quatY * ball.quatY + ball.quatZ * ball.quatZ);
  if (qLen > 1e-8f) {
    float invLen = 1.0f / qLen;
    ball.quatW *= invLen;
    ball.quatX *= invLen;
    ball.quatY *= invLen;
    ball.quatZ *= invLen;
  }
}

// ============================================================================
// computeGridDimensions - bounding box, grid origin, dimensions, and cell count
// ============================================================================
/**
 * @brief Compute grid dimensions and origin from domain parameters.
 * @param grid    GridData whose cellSize, origin*, gridDim*, numCells will be set.
 * @param params  SimParams providing slope geometry, domainLength, and cellSize.
 */
void computeGridDimensions(GridData& grid, const SimParams& params) {
  float cs   = params.slopeCos;
  float H    = params.slopeHeight;
  float L    = params.domainLength;
  float cell = params.cellSize;

  float margin = 1.0f;
  float minX = -margin;
  float maxX = L * cs + margin;
  float minY = -margin;
  float maxY = H / cs + margin;
  float patchHalfZ = params.spawnWidthZ * 0.5f + margin; // half spawn width + motion margin
  float minZ = -patchHalfZ - margin;
  float maxZ =  patchHalfZ + margin;

  grid.cellSize = cell;

  grid.originX  = minX;
  grid.originY  = minY;
  grid.originZ  = minZ;

  grid.gridDimX = (int)ceilf((maxX - minX) / cell);
  grid.gridDimY = (int)ceilf((maxY - minY) / cell);
  grid.gridDimZ = (int)ceilf((maxZ - minZ) / cell);

  // No capping here — each platform's allocateGrid() decides its own limits.
  // GPU uses compact hashing (hashTableSize << numCells) so large dims are fine.
  // CPU allocateGrid() clamps dims down to stay within RAM budget.

  grid.numCells = grid.gridDimX * grid.gridDimY * grid.gridDimZ;
}

// ============================================================================
// updateCoreMassAfterCapture - momentum-conserving core update after capture
// ============================================================================
/**
 * @brief Update snowball core mass/radius/velocity after particle capture.
 *        Applies momentum conservation: v_new = v_old * (m_old / m_new).
 * @param ball          SnowballState to update in place.
 * @param capturedCount Number of newly captured particles.
 * @param capturedMass  Total mass of captured particles.
 * @param snowDensity   Snow density for radius recalculation (kg/m³).
 */
void updateCoreMassAfterCapture(SnowballState& ball, int capturedCount, float capturedMass, float snowDensity) {
  if (capturedCount <= 0) return;

  float mOld = ball.mass;
  ball.mass += capturedMass;
  ball.capturedParticleCount += capturedCount;
  
  float scale = mOld / ball.mass;   // < 1 → ball slows
  ball.velX *= scale;
  ball.velY *= scale;
  ball.velZ *= scale;
  
  ball.radius = cbrtf(3.0f * ball.mass / (4.0f * (float)M_PI * snowDensity));
}

// ============================================================================
// printSimulationSummary - print final summary
// ============================================================================
/**
 * @brief Print final simulation summary (ball state, quaternion, angular velocity).
 * @param ball    Final snowball state.
 * @param shellN  Final number of shell particles.
 */
void printSimulationSummary(const SnowballState& ball, int shellN) {
  printf("Final ball: pos=(%.2f, %.2f, %.2f)\n            mass=%.3f\n            radius=%.3f\n            shell=%d\n",
         ball.posX, ball.posY, ball.posZ, ball.mass, ball.radius, shellN);
  printf("            quat=(%.4f, %.4f, %.4f, %.4f)\n            omega=(%.4f, %.4f, %.4f)\n\n",
         ball.quatW, ball.quatX, ball.quatY, ball.quatZ, ball.omegaX, ball.omegaY, ball.omegaZ);
}

// ============================================================================
// setParamsFromCLI - override SimParams fields from CLI arguments
// ============================================================================
/**
 * @brief Override SimParams fields from CLI arguments (if provided).
 *        Uses argInt/argFloat to parse each parameter, with current value as default.
 * @param params  SimParams pointer to update in place.
 * @param argc     CLI argument count from main().
 * @param argv     CLI argument vector from main().
 */
void setParamsFromCLI(SimParams* params, int argc, char** argv) {
  params->numParticles     = argInt  (argc, argv, "--particles",     params->numParticles);
  params->gravity          = argFloat(argc, argv, "--gravity",       params->gravity);
  //params->dt               = argFloat(argc, argv, "--dt",            params->dt);
  params->damping          = argFloat(argc, argv, "--damping",       params->damping);
  params->stiffness        = argFloat(argc, argv, "--stiffness",     params->stiffness);
  params->collisionDamping = argFloat(argc, argv, "--coll-damp",     params->collisionDamping);
  params->cohesion         = argFloat(argc, argv, "--cohesion",      params->cohesion);
  params->cohesionRadius   = argFloat(argc, argv, "--cohesion-rad",  params->cohesionRadius);
  params->particleRadius   = argFloat(argc, argv, "--part-radius",   params->particleRadius);
  params->particleMass     = argFloat(argc, argv, "--part-mass",     params->particleMass);
  params->restitution      = argFloat(argc, argv, "--restitution",   params->restitution);
  params->friction         = argFloat(argc, argv, "--friction",      params->friction);
  params->slopeAngleDeg    = argFloat(argc, argv, "--slope",         params->slopeAngleDeg);
  params->slopeHeight      = argFloat(argc, argv, "--slope-height",  params->slopeHeight);
  params->cellSize         = argFloat(argc, argv, "--cell-size",     params->cellSize);
  params->stickK0          = argFloat(argc, argv, "--stick-k0",      params->stickK0);
  params->stickK1          = argFloat(argc, argv, "--stick-k1",      params->stickK1);
  params->stickK2          = argFloat(argc, argv, "--stick-k2",      params->stickK2);
  params->stickRadiusBoost = argFloat(argc, argv, "--stick-rboost",  params->stickRadiusBoost);
  params->particleFriction = argFloat(argc, argv, "--part-friction", params->particleFriction);
  params->wetnessMin       = argFloat(argc, argv, "--wetness-min",   params->wetnessMin);
  params->wetnessMax       = argFloat(argc, argv, "--wetness-max",   params->wetnessMax);
  params->snowDensity      = argFloat(argc, argv, "--snow-density",  params->snowDensity);
  params->domainLength     = argFloat(argc, argv, "--domain-length", params->domainLength);
  params->spawnLengthS     = argFloat(argc, argv, "--spawn-length",  params->spawnLengthS);
  params->spawnWidthZ      = argFloat(argc, argv, "--spawn-width",   params->spawnWidthZ);
  params->spawnStartS      = argFloat(argc, argv, "--spawn-start",   params->spawnStartS);
  params->spawnThicknessN  = argFloat(argc, argv, "--spawn-thick",   params->spawnThicknessN);
  params->shellTetherK     = argFloat(argc, argv, "--tether-k",      params->shellTetherK);
  params->shellTetherDamp  = argFloat(argc, argv, "--tether-damp",   params->shellTetherDamp);
  params->logitClamp       = argFloat(argc, argv, "--logit-clamp",   params->logitClamp);
  params->maxCapturePerFrm = argInt  (argc, argv, "--max-capture",   params->maxCapturePerFrm);
  params->rollingDrag      = argFloat(argc, argv, "--rolling-drag",  params->rollingDrag);
  params->aeroCoeff        = argFloat(argc, argv, "--aero-coeff",    params->aeroCoeff);

  params->logInterval      = argInt  (argc, argv, "--log-interval",  params->logInterval);
  params->traceInterval    = argInt  (argc, argv, "--trace-interval",params->traceInterval);
  
  bool noLog               = argBool(argc, argv, "--no-log");
  bool noTrace             = argBool(argc, argv, "--no-trace");
  bool noCohesion          = argBool(argc, argv, "--no-cohesion");
  
  if (noLog)      params->logInterval   = 0;
  if (noTrace)    params->traceInterval = 0;
  if (noCohesion) params->cohesion      = 0.0f;
  
  // Precompute slope trigonometry
  float slopeRad   = params->slopeAngleDeg * (float)M_PI / 180.0f;
  params->slopeSin = sinf(slopeRad);
  params->slopeCos = cosf(slopeRad);
}

// ============================================================================
// printHelp - display CLI usage information
// ============================================================================
/**
 * @brief Print CLI usage information to stdout.
 *        Lists all available command-line options with descriptions and defaults.
 */
void printHelp(void) {
  printf("Usage: AngrySanta.exe [options]\n");
  printf("Options:\n");
  printf("  --particles <int>       Number of snowpack particles (default: 1000000)\n");
  printf("  --gravity <float>       Gravity magnitude in m/s² (default: 9.81)\n");
  printf("  --damping <float>       Linear velocity damping coefficient (default: 0.05)\n");
  printf("  --stiffness <float>     Collision penalty stiffness k_pen (default: 30000)\n");
  printf("  --coll-damp <float>     Collision damping k_damp (default: 50)\n");
  printf("  --cohesion <float>      Cohesion strength k_coh (default: 72450, set to 0 to disable)\n");
  printf("  --cohesion-rad <float>  Cohesion cutoff radius r_cut in meters (default: 0.06)\n");
  printf("  --part-radius <float>   Initial particle radius in meters (default: 0.02)\n");
  printf("  --part-mass <float>     Initial particle mass in kg (default: derived from radius and snow density)\n");
  printf("  --restitution <float>   Coefficient of restitution for collisions (default: 0.15)\n");
  printf("  --friction <float>      Coulomb friction coefficient for ground contact (default: 0.14)\n");
  printf("  --slope <float>         Slope angle in degrees (default: 30)\n");
  printf("  --slope-height <float>  Plane offset height in meters (default: 30)\n");
  printf("  --cell-size <float>     Spatial grid cell size in meters (default: spawn radius * 4)\n");
  printf("  --stick-k0 <float>      Base logit for sticking probability (default: 1.0)\n");
  printf("  --stick-k1 <float>      Wetness contribution to logit (default: 7.11)\n");
  printf("  --stick-k2 <float>      Velocity penalty in logit per m/s of relative velocity (default: 1.1)\n");
  printf("  --stick-rboost <float>  Logit boost per meter of ball radius (default: 7.5)\n");
  printf("  --part-friction <float> Inter-particle Coulomb friction coefficient (default: 0.04)\n");
  printf("  --wetness-min <float>   Minimum initial wetness (default: 0.0)\n");
  printf("  --wetness-max <float>   Maximum initial wetness (default: 1.0)\n");
  printf("  --snow-density <float>  Snow density in kg/m³ for mass/r adius calculation (default: 200)\n");
  printf("  --domain-length <float> Along-slope domain length in meters (default: 150)\n");
  printf("  --spawn-length <float>  Along-slope length of initial particle patch in meters (default: 1000)\n");
  printf("  --spawn-width <float>   Cross-slope width of initial particle patch in meters (default: 10)\n");
  printf("  --spawn-start <float>   Along-slope start position of particle patch in meters (default: 0)\n");
  printf("  --spawn-thick <float>   Vertical thickness of initial particle patch in meters (default: 0)\n");
  printf("  --tether-k <float>      Shell tether stiffness (default: 2717)\n");
  printf("  --tether-damp <float>   Shell tether damping (default: 50)\n");
  printf("  --logit-clamp <float>   Clamp for stick logit to prevent overflow (default: 15)\n");
  printf("  --max-capture <int>     Max particles captured per frame (default: 150)\n");
  printf("  --rolling-drag <float>  Rolling drag coefficient (default: 0.15)\n");
  printf("  --aero-coeff <float>    Aerodynamic drag coefficient (default: 0.3)\n");
  printf("  --log-interval <int>    Frames between log outputs (default: 0, disabled)\n");
  printf("  --trace-interval <int>  Frames between trace outputs (default: 0, disabled)\n");
  printf("  --no-log                Disable logging (sets log-interval to 0)\n");
  printf("  --no-trace              Disable trace output (sets trace-interval to 0)\n");
  printf("  --no-cohesion           Disable cohesion (sets cohesion to 0)\n");
  printf("  --help                  Show this help message and exit\n");
  printf("\n");
  printf("Only for the GPU version:\n");
  printf("  --render                Enable real-time rendering (default: disabled)\n");
}

// ============================================================================
// printConfig - print simulation configuration summary to stdout
// ============================================================================
/**
 * @brief Print simulation configuration summary to stdout.
 *        Includes all SimParams fields in a readable format.
 * @param params  SimParams to summarize.
 */
void printConfig(const SimParams& params) {
  printf("Snowpack  : %d particles (%.1fm x %.1fm, start=%.1fm, thick=%.2fm)\n",
        params.numParticles, params.spawnLengthS, params.spawnWidthZ,
        params.spawnStartS, params.spawnThicknessN);
  printf("MaxCapture: %d per frame\n", params.maxCapturePerFrm);
  printf("Physics   : g=%.1f m/s^2, dt=%.4fs, damping=%.2f\n", params.gravity, params.dt, params.damping);
  printf("Collision : stiffness=%.1f, coll. damping=%.1f, restitution=%.2f\n", params.stiffness, params.collisionDamping, params.restitution);
  printf("            friction=%.2f, particle friction=%.2f\n", params.friction, params.particleFriction);
  printf("Cohesion  : %s (k=%.1f, r=%.3f)\n",
        (params.cohesion == 0.0f) ? "OFF" : "ON",
        params.cohesion, params.cohesionRadius);
  printf("Particle  : radius=%.2fm, mass=%.2fKg, density=%.1f kg/m^3\n", params.particleRadius, params.particleMass, params.snowDensity);
  printf("Terrain   : slope=%.1f°, slope height=%.1fm, domain len=%.1fm\n", params.slopeAngleDeg, params.slopeHeight, params.domainLength);
  printf("Stick MVP : k0=%.2f, k1=%.2f, k2=%.2f, rBoost=%.2f, logitClamp=%.1f\n",
        params.stickK0, params.stickK1, params.stickK2,
        params.stickRadiusBoost, params.logitClamp);
  printf("Wetness   : [%.2f, %.2f]\n", params.wetnessMin, params.wetnessMax);
  printf("Ball      : rolling drag=%.2f, aero coeff=%.2f\n", params.rollingDrag, params.aeroCoeff);
  printf("Shell     : tether K=%.1f, tether damp=%.1f\n", params.shellTetherK, params.shellTetherDamp);
  printf("            cell size=%.4fm\n", params.cellSize);
  printf("\n");
  printf("Trace int : %d frames%s\n", params.traceInterval, params.traceInterval == 0 ? " (disabled)" : "");
  printf("Log every : %d frames%s\n", params.logInterval, params.logInterval == 0 ? " (disabled)" : "");
  printf("\n");
}