#ifndef TYPES_H
  #define TYPES_H

  #include <cstdint>
  #include <cstddef> // mandatory on LINUX

  #ifndef M_PI
    #define M_PI 3.14159265358979323846f
  #endif

  // ============================================================================
  // Common types and structs used across multiple modules
  // ============================================================================

  enum ShellParticleState : int {
    INACTIVE = -1,
    ACTIVE   =  1
  };

  enum SnowpackParticleState : int {
    AVAILABLE,
    CONSUMED
  };

  // Do NOT declare extern here: with MSVC/CUDA separate compilation it causes
  // "named symbol not found" at runtime.  Kernels that need SimParams access
  // it directly from the __constant__ variable in their own TU.
  struct SimParams {

    // Physics
    float gravity;             // m/s²  (magnitude, applied along slope / -Y)
    float dt;                  // timestep (s)
    float damping;             // linear velocity damping coefficient (k_d)

    // Soft-sphere collision
    float stiffness;           // penalty stiffness (k_pen)
    float collisionDamping;    // collision damping (k_damp)

    // Cohesion (short-range attraction)
    float cohesion;            // cohesion strength (k_coh)
    float cohesionRadius;      // cutoff radius for cohesion (r_cut)

    // Particle geometry
    float particleRadius;      // uniform initial radius (r_p)
    float particleMass;        // uniform mass (derived: 4/3 π r³ ρ)

    // Ground interaction
    float restitution;         // coefficient of restitution (e_ground)
    float friction;            // Coulomb friction coefficient (mu_ground)

    // Terrain
    float slopeAngleDeg;       // slope angle in degrees
    float slopeSin;            // precomputed sin(angle)
    float slopeCos;            // precomputed cos(angle)
    float slopeHeight;         // plane offset: sin(θ)*x + cos(θ)*y = slopeHeight

    // Spatial grid
    float cellSize;            // uniform grid cell size (m), default = 4 * particleRadius

    // Probabilistic sticking (MVP avalanche model)
    float stickK0;             // base logit for sticking probability
    float stickK1;             // wetness contribution to logit
    float stickK2;             // velocity penalty in logit (per m/s)
    float stickRadiusBoost;    // logit boost per metre of ball radius (avalanche feedback)
    float particleFriction;    // inter-particle tangential friction (Coulomb)
    float wetnessMin;          // minimum initial wetness [0,1]
    float wetnessMax;          // maximum initial wetness [0,1]

    float snowDensity;         // snow density for radius calc (kg/m³)

    // Trace
    int traceInterval;         // frames between trace snapshots (0 = disabled)

    // Domain
    float domainLength;        // max along-slope distance before particle is killed
    int numParticles;          // snowpack particle count (static terrain)
    int logInterval;           // frames between log outputs (0 = no log)

    // ---- Snowpack spawn geometry ----
    float spawnLengthS;        // patch length along slope (m)
    float spawnWidthZ;         // patch width lateral Z (m)
    float spawnStartS;         // start offset along slope (m)
    float spawnThicknessN;     // thickness along slope normal (m), 0 = single layer

    // ---- Shell physics ----
    float shellTetherK;        // tether spring stiffness to core (N/m)
    float shellTetherDamp;     // tether damping coefficient

    // ---- Avalanche stability ----
    float logitClamp;          // max absolute logit value for sticking sigmoid
    int   maxCapturePerFrm;    // max particles to capture per frame (0 = unlimited)

    // ---- Rigid-body drag ----
    float rollingDrag;         // Physically represents rolling friction + surface deformation losses
    float aeroCoeff;           // Aerodynamic drag coefficient
  };

  struct SnowballState {
    float posX, posY, posZ;    // center of mass
    float velX, velY, velZ;    // velocity
    float radius;              // current radius
    float mass;                // current total mass
    int capturedParticleCount; // cumulative number of particles captured into the ball

    // Orientation (unit quaternion, Hamilton convention: w + xi + yj + zk)
    float quatW, quatX, quatY, quatZ;

    // Angular velocity (rad/s, world frame)
    float omegaX, omegaY, omegaZ;
  };

  struct ParticleSystem  {

    // Positions
    float *posX;
    float *posY;
    float *posZ;

    // Velocities
    float *velX;
    float *velY;
    float *velZ;

    // Forces / accelerations (accumulated per frame)
    float *forceX;
    float *forceY;
    float *forceZ;

    // Per-particle attributes
    float *mass;
    float *radius;
    ShellParticleState *state;

    // Per-particle adhesion / wetness [0,1]
    float *wetness;

    // Body-frame tether anchors for shell particles.
    float *attachLocalX;
    float *attachLocalY;
    float *attachLocalZ;

    int capacity; // allocated size
  };

  // Static snowpack - particles on the slope surface, consumed by capture
  struct Snowpack {
    float *posX, *posY, *posZ; // world-space position
    float *radius;             // per-particle radius
    float *mass;               // per-particle mass
    float *wetness;            // adhesion factor [0,1]
    int    count;              // total allocated
    SnowpackParticleState *alive;
  };

  struct GridData {

    // Per-particle arrays (size = N)
    uint32_t *cellHash;        // hash of the cell each particle belongs to
    uint32_t *particleIndex;   // original particle index (pre-sort order)

    // Double-buffer for CUB radix sort (output arrays)
    uint32_t *cellHashAlt;
    uint32_t *particleIndexAlt;

    // CUB radix sort temp storage
    void   *d_sortTemp;
    size_t  sortTempBytes;

    // Per-cell arrays (size = numCells)
    int *cellStart;            // first index in sorted array (-1 = empty)
    int *cellEnd;              // exclusive end index in sorted array

    // Grid configuration (computed at init, read-only during sim)
    int   numCells;
    int   gridDimX, gridDimY, gridDimZ;
    float cellSize;
    float originX, originY, originZ;

    int   hashTableSize;
    int   hashTableMask;       // hashTableSize - 1, for fast bitwise AND modulo
  };

  struct KernelTimings {
    float captureMs;           // snowpack → shell capture
    float shellForcesMs;       // forcesTether: gravity + damping + tether (FUSED)
    float shellGridMs;         // spatial grid build for shell
    float shellCollisionMs;    // neighbor collision + cohesion in shell
    float shellIntegrateMs;    // intGround: integration + ground collision (FUSED)
    float shellFusedMs;        // mega-fused kernel (forces+collision+integrate, small N)
    float coreUpdateMs;        // rigid-body snowball update
    float totalFrameMs;
  };
#endif // TYPES_H