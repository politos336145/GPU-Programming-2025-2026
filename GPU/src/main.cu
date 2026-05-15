// ============================================================================
// Angry Santa - Snowball Simulation (Snowpack + Shell + Rigid Core)
// Main entry point and simulation loop
//
// Per-frame pipeline:
//   1. Capture from snowpack → append to shell
//   2. Core mass / radius update after capture
//   3. Fused Forces+Tether on simStream
//   4. Grid build on gridStream (fork/join via events)
//   5. Neighbor collision (simStream, after grid done)
//   6. Fused Integrate+Ground (simStream)
//   7. Rigid-body core update
// ============================================================================

#include "include/grid.cuh"
#include "include/helpers.cuh"
#include "include/simulation.cuh"
#include "include/snowpack.cuh"
#include "../../shared/include/cli.h"
#include "../../shared/include/global.h"
#include "../../shared/include/logging.h"
#include "../../shared/include/profiling.h"

#ifdef ENABLE_VULKAN
  #include "renderers/vulkan_renderer.h"
#endif

// NVTX ranges for Nsight Systems timeline annotation
#ifdef ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>

  #define NVTX_PUSH(name)  nvtxRangePushA(name)
  #define NVTX_POP()       nvtxRangePop()
#else
  #define NVTX_PUSH(name)  ((void)0)
  #define NVTX_POP()       ((void)0)
#endif

#include <algorithm> // mandatory on LINUX
#include <chrono>

Logger logger;
SimParams params = defaultParams();
std::string timestamp;

// Print header with GPU info
inline void _printHeader(bool renderEnabled) {
  int device;
  cudaDeviceProp prop;

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  printf("======================================================================\n");
  printf("===      Angry Santa - Snowball and Avalanche Simulation           ===\n");
  printf("===      GPU: %s (SM %d.%d - VRAM %d MB)%s===\n", prop.name, prop.major, prop.minor, (int)(prop.totalGlobalMem / (1024 * 1024)), std::string(70 - strlen(prop.name) - 41, ' ').c_str());
  printf("===           Shared Memory per Block: %zu KB%s===\n", prop.sharedMemPerBlock / 1024, std::string(70 - strlen((std::to_string(prop.sharedMemPerBlock / 1024)).c_str()) - 45, ' ').c_str());
  printf("===           Registers per Block: %d%s===\n", prop.regsPerBlock, std::string(70 - strlen((std::to_string(prop.regsPerBlock)).c_str()) - 38, ' ').c_str());
  printf("===           Warp size: %d%s===\n", prop.warpSize, std::string(70 - strlen((std::to_string(prop.warpSize)).c_str()) - 28, ' ').c_str());
  printf("===           Max threads per block: %d%s===\n", prop.maxThreadsPerBlock, std::string(70 - strlen((std::to_string(prop.maxThreadsPerBlock)).c_str()) - 40, ' ').c_str());
  printf("===           Number of multiprocessors: %d%s===\n", prop.multiProcessorCount, std::string(70 - strlen((std::to_string(prop.multiProcessorCount)).c_str()) - 44, ' ').c_str());
  printf("===           Max threads per multiprocessor: %d%s===\n", prop.maxThreadsPerMultiProcessor, std::string(70 - strlen((std::to_string(prop.maxThreadsPerMultiProcessor)).c_str()) - 49, ' ').c_str());
  printf("===           Clock rate: %.2f MHz%s===\n", prop.clockRate / 1000.0, std::string(70 - strlen((std::to_string(prop.clockRate / 1000.0)).c_str()) - 29, ' ').c_str());
  printf("===                                                                ===\n");
  #ifdef ENABLE_VULKAN
    if (renderEnabled)
      printf("===      Rendering: Vulkan enabled                                 ===\n");
    else
      printf("===      Rendering: disabled (use --render flag to enable)         ===\n");
  #else
    printf("===      Rendering: Vulkan unsupported                             ===\n");
  #endif
  printf("======================================================================\n");
}

// Print current VRAM usage
inline void _printVRAMUsage(void) {
  size_t freeMem = 0, totalMem = 0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("[VRAM] Free: %.1f MB / %.1f MB (used: %.1f MB)\n",
          freeMem / (1024.0 * 1024.0),
          totalMem / (1024.0 * 1024.0),
          (totalMem - freeMem) / (1024.0 * 1024.0));
}

// ============================================================================
// Main
// ============================================================================
/**
 * @brief Main entry point: parse CLI, set up GPU resources, run simulation loop.
 * 
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * 
 * @return 0 on success.
 */
int main(int argc, char** argv) {
  bool renderEnabled = argBool(argc, argv, "--render");
  _printHeader(renderEnabled);

  if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    printHelp();
    return 0;
  }
  
  auto progStart = std::chrono::high_resolution_clock::now();

  setParamsFromCLI(&params, argc, argv);
    
  #ifdef ENABLE_VULKAN
    if (renderEnabled) {
      RenderConfig rcfg;
      if (initRenderer(rcfg, 256, params) != 0) {
        fprintf(stderr, "Renderer init failed, falling back to headless\n");
        renderEnabled = false;
      }
    }
  #endif

  // =========================================================================
  // Simulation loop - wraps alloc → sim → idle so that pressing
  // START in the post-sim panel relaunches with the user-edited parameters.
  // =========================================================================
  
  bool isFirstRun = true;
  bool windowClosed = false;
  bool doRestart = false;

  do {
    KernelTimings kt = {};
    ProfilingStats profStats;
    profStats.reset();

    int shellN = 0;  
    timestamp = newTimestamp();
    
    printConfig(params);
    copyParamsToGPU(params);
        
    if (!isFirstRun) printf("\n[Restart] New run\n");
    if (params.logInterval > 0) logger.openCSV((timestamp + "_log_GPU.csv").c_str());

    // -----------------------------------------------------------------------
    // System Preparation
    // -----------------------------------------------------------------------

    // Snowpack (static terrain particles)
    Snowpack snowpack;
    allocateSnowpack(snowpack, params.numParticles);
    launchInitSnowpack(snowpack, rand() % 100 + 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    sortSnowpackByPosX(snowpack); // Sort snowpack by posX for range-limited capture

    // Copy sorted posX to pinned host memory for per-frame binary search
    float *h_snowpackPosX = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_snowpackPosX, snowpack.count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h_snowpackPosX, snowpack.posX, snowpack.count * sizeof(float), cudaMemcpyDeviceToHost));

    // Snowball core (rigid body)
    SnowballState ball;
    initSnowball(ball, params);

    // Shell (dynamic particles around ball)
    ParticleSystem shell;
    int shellCap = 256;  // initial small capacity, grows dynamically
    allocateParticleSystem(shell, shellCap);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Spatial grid for shell neighbor search
    GridData grid;
    allocateGrid(grid, shell.capacity, params);

    #ifdef ENABLE_VULKAN
      if (renderEnabled) {
        if (isFirstRun)
          notifySimStarted(params);
        else {
          // Restart iteration: params were already captured from the post-sim panel via getRendererSimParams()
          // Just sync GPU constant memory and make sure the VBO is the right size for the (possibly new) shellCap
          resizeRendererVBO(shell.capacity);
          notifySimStarted(params);
        }
      }
    #endif

    // Device counters for capture - packed into contiguous 8 bytes for single memset/memcpy
    // Layout: [int count (4B)][float mass (4B)]
    char *d_captureBuf;
    CUDA_CHECK(cudaMalloc(&d_captureBuf, 8));
    int   *d_captureCount = reinterpret_cast<int*>(d_captureBuf);
    float *d_captureMass  = reinterpret_cast<float*>(d_captureBuf + 4);

    // Pinned host memory for capture counter readback
    // Using cudaMallocHost avoids the hidden cudaDeviceSynchronize when cudaMemcpyAsync targets pageable (stack) memory
    char  *h_captureBuf;
    CUDA_CHECK(cudaMallocHost(&h_captureBuf, 8));
    int   *h_captureCount = reinterpret_cast<int*>(h_captureBuf);
    float *h_captureMass  = reinterpret_cast<float*>(h_captureBuf + 4);

    // Pinned memory for trace snapshots (shell particles)
    float *h_posX = nullptr, *h_posY = nullptr, *h_posZ = nullptr;
    float *h_velX = nullptr, *h_velY = nullptr, *h_velZ = nullptr;
    float *h_forceX = nullptr, *h_forceY = nullptr, *h_forceZ = nullptr;
    float *h_attachX = nullptr, *h_attachY = nullptr, *h_attachZ = nullptr;
    float *h_mass = nullptr;
    float *h_radius = nullptr;
    ShellParticleState *h_state = nullptr;
    float *h_wetness = nullptr;
    
    cudaStream_t traceStream = nullptr;
    if (params.traceInterval > 0) {
      CUDA_CHECK(cudaMallocHost(&h_posX, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_posY, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_posZ, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_velX, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_velY, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_velZ, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_forceX, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_forceY, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_forceZ, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_attachX, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_attachY, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_attachZ, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_mass, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_radius, shell.capacity * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&h_state, shell.capacity * sizeof(ShellParticleState)));
      CUDA_CHECK(cudaMallocHost(&h_wetness, shell.capacity * sizeof(float)));

      CUDA_CHECK(cudaStreamCreateWithFlags(&traceStream, cudaStreamNonBlocking));
      printf("Trace: pinned buffers + async stream allocated (%.2f MB)\n", (shell.capacity * 15 * sizeof(float) / (1024.0f * 1024.0f) +
                                                                           (shell.capacity * sizeof(ShellParticleState) / (1024.0f * 1024.0f))));
    }

    // -----------------------------------------------------------------------
    // CUDA Streams - physics pipeline
    //   simStream  : Forces → Tether → (wait grid) → Collision → Integrate → Ground
    //   gridStream : Grid build (hash + CUB sort + cell ranges)
    // Fork/join via lightweight events (timing disabled for sync-only use).
    // -----------------------------------------------------------------------
    cudaStream_t simStream = nullptr;
    cudaStream_t gridStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&simStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&gridStream, cudaStreamNonBlocking));

    cudaEvent_t evFork = nullptr;
    cudaEvent_t evGridDone = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&evFork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evGridDone, cudaEventDisableTiming));

    printf("Streams: simStream + gridStream (non-blocking) created\n");

    // Deferred capture readback - event to signal D2H completion
    cudaEvent_t evCaptureReady = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&evCaptureReady, cudaEventDisableTiming));
    bool captureReadbackPending = false;

    // CUDA event timers
    enum TimerIdx { T_FRAME = 0, T_CAPTURE, T_FORCES_TETHER, T_GRID, T_NEIGHBOR, T_INTEGRATE_GROUND, T_FUSED, T_COUNT };
    static constexpr int NUM_TIMERS = T_COUNT;
    cudaEvent_t evStart[NUM_TIMERS], evEnd[NUM_TIMERS];
    for (int i = 0; i < NUM_TIMERS; i++) {
      CUDA_CHECK(cudaEventCreate(&evStart[i]));
      CUDA_CHECK(cudaEventCreate(&evEnd[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    _printVRAMUsage();

    // -----------------------------------------------------------------------
    // Simulation loop
    // -----------------------------------------------------------------------
    auto wallStart = std::chrono::high_resolution_clock::now();
    int totalFrames = 0;
    constexpr int TIMING_INTERVAL = 64;

    for (int frame = 0; ; frame++) {
      NVTX_PUSH("Frame");

        bool profileThisFrame = renderEnabled ||
                                (params.logInterval > 0 && frame % params.logInterval == 0) ||
                                frame % TIMING_INTERVAL == 0;

        // ---- Process PREVIOUS frame's capture readback (deferred D2H) ----
        // By now, the entire previous frame's physics pipeline ran on simStream
        // AFTER the D2H memcpy, so evCaptureReady is guaranteed signaled.
        // Why defer to the next frame?
        // 1) The capture → mass update → radius feedback loop is self-limiting and stable
        // 2) The 1-frame latency is negligible in practice
        // 3) This also allows us to batch the D2H with the next frame's capture kernel, eliminating an extra cudaEventRecord + cudaEventSynchronize pair per frame.
        if (captureReadbackPending) {
          CUDA_CHECK(cudaEventSynchronize(evCaptureReady));
          int   h_newCount = *h_captureCount;
          float h_newMass  = *h_captureMass;
          int maxCapWrite = params.maxCapturePerFrm > 0 ? params.maxCapturePerFrm : h_newCount;
          int actualNew = (std::min)(h_newCount, (std::min)(maxCapWrite, shell.capacity - shellN));
          shellN += actualNew;
          // Scale captured mass proportionally to particles actually accepted.
          // d_captureMass accumulates ALL particles that passed P_stick on the GPU,
          // including those later discarded by maxCapturePerFrm or capacity.
          // If actualNew < h_newCount we must credit only the accepted fraction
          // to avoid artificially inflating ball.radius and triggering a runaway
          // capture feedback loop via stickRadiusBoost.
          float scaledMass = (h_newCount > 0) ? h_newMass * ((float)actualNew / (float)h_newCount) : 0.0f;
          updateCoreMassAfterCapture(ball, actualNew, scaledMass, params.snowDensity);
          captureReadbackPending = false;
        }
        if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_FRAME], simStream));

        // 1. Capture from snowpack → append to shell
        NVTX_PUSH("Capture");

          {
            int maxCapWrite = params.maxCapturePerFrm > 0 ? params.maxCapturePerFrm : params.numParticles;
            int need = shellN + maxCapWrite;
            if (need > shell.capacity) {
              CUDA_CHECK(cudaDeviceSynchronize());
              int newCap = shell.capacity > 0 ? shell.capacity : 256;
              while (newCap < need)
                newCap *= 2;
              
              growShell(shell, shellN, newCap);
              resizeGrid(grid, newCap, params);

              if (params.traceInterval > 0) {
                cudaFreeHost(h_posX);    cudaFreeHost(h_posY);    cudaFreeHost(h_posZ);
                cudaFreeHost(h_velX);    cudaFreeHost(h_velY);    cudaFreeHost(h_velZ);
                cudaFreeHost(h_forceX);  cudaFreeHost(h_forceY);  cudaFreeHost(h_forceZ);
                cudaFreeHost(h_attachX); cudaFreeHost(h_attachY); cudaFreeHost(h_attachZ);
                cudaFreeHost(h_mass);
                cudaFreeHost(h_radius);
                cudaFreeHost(h_state);
                cudaFreeHost(h_wetness);

                CUDA_CHECK(cudaMallocHost(&h_posX,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_posY,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_posZ,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_velX,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_velY,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_velZ,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_forceX,  newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_forceY,  newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_forceZ,  newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_attachX, newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_attachY, newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_attachZ, newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_mass,    newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_radius,  newCap * sizeof(float)));
                CUDA_CHECK(cudaMallocHost(&h_state,   newCap * sizeof(ShellParticleState)));
                CUDA_CHECK(cudaMallocHost(&h_wetness, newCap * sizeof(float)));
              }
              
              #ifdef ENABLE_VULKAN
                if (renderEnabled) resizeRendererVBO(newCap);
              #endif
            }
          }

          if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_CAPTURE], simStream));
          {
            CUDA_CHECK(cudaMemsetAsync(d_captureBuf, 0, 8, simStream));

            // Binary search sorted snowpack posX for the narrow range around the ball.
            // Only launch threads for particles whose X is within capture distance.
            float captureR = ball.radius + params.particleRadius + 0.3f;
            float loX = ball.posX - captureR;
            float hiX = ball.posX + captureR;
            int spLo = (int)(std::lower_bound(h_snowpackPosX, h_snowpackPosX + snowpack.count, loX) - h_snowpackPosX);
            int spHi = (int)(std::upper_bound(h_snowpackPosX, h_snowpackPosX + snowpack.count, hiX) - h_snowpackPosX);

            launchCaptureFromSnowpack(
              snowpack, shell, shellN, ball,
              d_captureCount, d_captureMass,
              frame, shell.capacity, spLo, spHi - spLo, simStream);
          }
          if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_CAPTURE], simStream));

        NVTX_POP();  // Capture

        {
          // Kick off D2H into pinned memory and record event - NO host blocking.
          // The result is processed at the top of the NEXT frame (1-frame latency,
          // physically negligible: newly captured particles join physics 1 dt later).
          CUDA_CHECK(cudaMemcpyAsync(h_captureBuf, d_captureBuf, 8, cudaMemcpyDeviceToHost, simStream));
          CUDA_CHECK(cudaEventRecord(evCaptureReady, simStream));
          captureReadbackPending = true;
        }

        // 2. Core mass / radius update - handled in deferred readback above

        // ====================================================================
        // Stages 3-7: Physics pipeline
        //
        // For small shell counts (≤ BF_COLLISION_THRESHOLD) the spatial grid
        // build (5-8 GPU commands) costs more in launch overhead than the
        // collision itself.  Switch to an O(N²) brute-force kernel that runs
        // as a single launch on simStream, eliminating the fork/join and all
        // grid infrastructure commands.
        //
        // For larger shells the grid-based pipeline provides O(N·k) scaling:
        //   simStream  : ForcesTether ────────────→ Collision → IntegrateGround
        //   gridStream :         ├── Grid build ──┘  (fork/join via events)
        // ====================================================================
        constexpr int BF_COLLISION_THRESHOLD = 2048;
        bool usedFusedKernel = false;
        
        {
          NVTX_PUSH("MultiStream");

            if (shellN > 0 && shellN <= BF_COLLISION_THRESHOLD) {
              // ---- Small-N path: mega-fused kernel ----
              // forceTether + bruteForceCollision + integrateGround in ONE launch.
              // Forces stay entirely in registers (no global memory round-trip).
              // Eliminates grid build, fork/join, and 2 extra kernel launches.
              usedFusedKernel = true;
              if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_FUSED], simStream));

              NVTX_PUSH("ShellPhysicsFused");
                launchShellPhysicsFused(shell, ball, ball.radius, shellN, simStream);
              NVTX_POP();

              if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_FUSED], simStream));
            } else if (shellN > BF_COLLISION_THRESHOLD) {
              // ---- Large-N path: grid-accelerated collision (3 separate kernels) ----

              // 3. Fused Forces+Tether on simStream
              NVTX_PUSH("ForcesTether");
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_FORCES_TETHER], simStream));
                launchForcesTether(shell, ball, shellN, simStream);
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_FORCES_TETHER], simStream));
              NVTX_POP();

              // Fork: simStream → gridStream
              CUDA_CHECK(cudaEventRecord(evFork, simStream));
              CUDA_CHECK(cudaStreamWaitEvent(gridStream, evFork, 0));

              // 4. Grid build on gridStream
              NVTX_PUSH("Grid");
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_GRID], gridStream));
                launchBuildGrid(grid, shell, shellN, gridStream);
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_GRID], gridStream));
              NVTX_POP();

              // Join: gridStream → simStream
              CUDA_CHECK(cudaEventRecord(evGridDone, gridStream));
              CUDA_CHECK(cudaStreamWaitEvent(simStream, evGridDone, 0));

              // 5. Neighbor collision (simStream, after grid done)
              NVTX_PUSH("Collision");
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_NEIGHBOR], simStream));
                launchNeighborCollision(shell, grid, shellN, simStream);
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_NEIGHBOR], simStream));
              NVTX_POP();

              // 6. Fused Integrate+Ground (simStream)
              NVTX_PUSH("IntegrateGround");
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evStart[T_INTEGRATE_GROUND], simStream));
                launchIntegrateGround(shell, shellN, simStream);
                if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_INTEGRATE_GROUND], simStream));
              NVTX_POP();
            }

          NVTX_POP();  // MultiStream
        }

        // 7. Rigid-body core update
        updateSnowball(ball, params);

        if (profileThisFrame) CUDA_CHECK(cudaEventRecord(evEnd[T_FRAME], simStream));
      
      NVTX_POP();  // Frame

      // ---- Termination: ball has reached the ground ----
      bool ballAtGround = (ball.posY <= ball.radius + 0.01f) && (ball.velY >= -0.01f);
      totalFrames = frame + 1;

      // ---- Deferred timing: only sync + read every TIMING_INTERVAL frames,
      //      or when logging / terminating / rendering.  In headless mode this
      //      eliminates the per-frame cudaEventSynchronize host stall. ----
      bool needSync = profileThisFrame || ballAtGround;
      if (needSync) {
        if (profileThisFrame) {
          CUDA_CHECK(cudaEventSynchronize(evEnd[T_FRAME]));

          // ---- Read per-kernel timings ----
          CUDA_CHECK(cudaEventElapsedTime(&kt.totalFrameMs, evStart[T_FRAME], evEnd[T_FRAME]));
          CUDA_CHECK(cudaEventElapsedTime(&kt.captureMs,    evStart[T_CAPTURE], evEnd[T_CAPTURE]));

          if (usedFusedKernel) {
            kt.shellForcesMs    = 0.0f;
            kt.shellGridMs      = 0.0f;
            kt.shellCollisionMs = 0.0f;
            kt.shellIntegrateMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kt.shellFusedMs, evStart[T_FUSED], evEnd[T_FUSED]));
          } else if (shellN > 0) {
            CUDA_CHECK(cudaEventElapsedTime(&kt.shellForcesMs,    evStart[T_FORCES_TETHER],    evEnd[T_FORCES_TETHER]));
            CUDA_CHECK(cudaEventElapsedTime(&kt.shellGridMs,      evStart[T_GRID],             evEnd[T_GRID]));
            CUDA_CHECK(cudaEventElapsedTime(&kt.shellCollisionMs, evStart[T_NEIGHBOR],         evEnd[T_NEIGHBOR]));
            CUDA_CHECK(cudaEventElapsedTime(&kt.shellIntegrateMs, evStart[T_INTEGRATE_GROUND], evEnd[T_INTEGRATE_GROUND]));
            kt.shellFusedMs = 0.0f;
          } else { // shellN == 0: no physics events recorded
            kt.shellForcesMs    = 0.0f;
            kt.shellGridMs      = 0.0f;
            kt.shellCollisionMs = 0.0f;
            kt.shellIntegrateMs = 0.0f;
            kt.shellFusedMs     = 0.0f;
          }

          kt.coreUpdateMs = 0.0f;
          profStats.accumulate(kt);
        } else
          CUDA_CHECK(cudaStreamSynchronize(simStream));
      }

      float frameMs = (needSync && profileThisFrame) ? kt.totalFrameMs : 0.0f;

      #ifdef ENABLE_VULKAN
        if (renderEnabled) {
          RenderTiming rt;
          renderFrame(shell, ball, shellN, rt);

          if (params.logInterval > 0 && (frame % params.logInterval == 0))
            setRendererTitle(frame, shellN, ball.capturedParticleCount, frameMs, rt.vboFillMs, rt.drawMs);

          if (shouldCloseRenderer()) {
            printf("\n[Renderer] Window closed by user at frame %d\n", frame);
            break;
          }
        }
      #endif

      // -----------------------------------------------------------------------
      // Logging
      // -----------------------------------------------------------------------
      if (params.logInterval > 0 &&
          (frame % params.logInterval == 0 || ballAtGround)) {
        FrameStats stats = {};
        stats.frame               = frame;
        stats.shellCount          = shellN;
        stats.snowpackAlive       = snowpack.count - ball.capturedParticleCount;
        stats.ballMass            = ball.mass;
        stats.ballRadius          = ball.radius;
        stats.ballPosX            = ball.posX;
        stats.ballPosY            = ball.posY;
        stats.ballPosZ            = ball.posZ;
        stats.ballVelX            = ball.velX;
        stats.ballVelY            = ball.velY;
        stats.ballVelZ            = ball.velZ;
        stats.frameTimeMs         = frameMs;
        stats.captureMs           = kt.captureMs;
        stats.shellForcesTetherMs = kt.shellForcesMs;
        stats.shellGridMs         = kt.shellGridMs;
        stats.shellCollisionMs    = kt.shellCollisionMs;
        stats.shellIntGroundMs    = kt.shellIntegrateMs;
        stats.coreUpdateMs        = kt.coreUpdateMs;

        logger.logFrame(stats);
        if (shellN > 0) printGridStats(grid, shellN, frame);
      }

      // -----------------------------------------------------------------------
      // Trace snapshot (shell particles)
      // -----------------------------------------------------------------------
      if (params.traceInterval > 0 &&
          frame % params.traceInterval == 0 &&
          shellN > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_posX, shell.posX, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_posY, shell.posY, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_posZ, shell.posZ, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_velX, shell.velX, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_velY, shell.velY, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_velZ, shell.velZ, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_forceX, shell.forceX, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_forceY, shell.forceY, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_forceZ, shell.forceZ, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_attachX, shell.attachLocalX, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_attachY, shell.attachLocalY, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_attachZ, shell.attachLocalZ, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_mass, shell.mass, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_radius, shell.radius, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_state, shell.state, shellN * sizeof(ShellParticleState), cudaMemcpyDeviceToHost, traceStream));
        CUDA_CHECK(cudaMemcpyAsync(h_wetness, shell.wetness, shellN * sizeof(float), cudaMemcpyDeviceToHost, traceStream));
        
        CUDA_CHECK(cudaStreamSynchronize(traceStream));

        char fname[64];
        snprintf(fname, sizeof(fname), "%s_%05d_trace_gpu.csv", timestamp.c_str(), frame);
        FILE *tf = fopen(fname, "w");
        if (tf) {
          fprintf(tf, "posX,posY,posZ,velX,velY,velZ,forceX,forceY,forceZ,attachX,attachY,attachZ,mass,radius,state,wetness\n");
          for (int i = 0; i < shellN; i++) {
            fprintf(tf, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f\n",
                    h_posX[i], h_posY[i], h_posZ[i],
                    h_velX[i], h_velY[i], h_velZ[i],
                    h_forceX[i], h_forceY[i], h_forceZ[i],
                    h_attachX[i], h_attachY[i], h_attachZ[i],
                    h_mass[i],
                    h_radius[i],
                    h_state[i],
                    h_wetness[i]);
          }
          fclose(tf);
          printf("  [Trace] wrote %s (%d shell particles)\n", fname, shellN);
        }
      }

      if (ballAtGround) break;
    }

    // ---- Flush any pending capture readback ----
    if (captureReadbackPending) {
      CUDA_CHECK(cudaEventSynchronize(evCaptureReady));
      int   h_newCount = *h_captureCount;
      float h_newMass  = *h_captureMass;
      int maxCapWrite = params.maxCapturePerFrm > 0 ? params.maxCapturePerFrm : h_newCount;
      int actualNew = (std::min)(h_newCount, (std::min)(maxCapWrite, shell.capacity - shellN));
      shellN += actualNew;
      float scaledMass = (h_newCount > 0) ? h_newMass * ((float)actualNew / (float)h_newCount) : 0.0f;
      updateCoreMassAfterCapture(ball, actualNew, scaledMass, params.snowDensity);
      captureReadbackPending = false;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto wallEnd = std::chrono::high_resolution_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
    double wallFps = (wallMs > 0.0) ? totalFrames * 1000.0 / wallMs : 0.0;

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("Wall-clock: %.1f ms total, %.1f FPS (%.3f ms/frame)\n", wallMs, wallFps, (totalFrames > 0) ? wallMs / totalFrames : 0.0);
    printSimulationSummary(ball, shellN);
    profStats.printSummary(shellN, snowpack.count, totalFrames);

    // -----------------------------------------------------------------------
    // Post-simulation idle loop - restart-aware
    // -----------------------------------------------------------------------
    windowClosed = false;
    #ifdef ENABLE_VULKAN
      if (renderEnabled && !shouldCloseRenderer()) {
        notifySimComplete();  // re-enables the param panel in the idle loop
        printf("\n[Renderer] Simulation complete - edit parameters and press START to restart, or close to exit.\n");

        char idleTitle[256];
        snprintf(idleTitle, sizeof(idleTitle), "Angry Santa - DONE (%d frames, shell %d) - edit params + START to restart", totalFrames, shellN);
        setRendererTitle(idleTitle);

        while (!shouldCloseRenderer()) {
          RenderTiming rt;
          renderFrame(shell, ball, shellN, rt);
          if (consumeStartPressed()) {
            getRendererSimParams(params);
            // Recompute derived trig from the (possibly edited) slopeAngleDeg
            float slopeRad   = params.slopeAngleDeg * (float)M_PI / 180.0f;
            params.slopeSin  = sinf(slopeRad);
            params.slopeCos  = cosf(slopeRad);
            doRestart = true;
            
            break;
          }
        }
        windowClosed = shouldCloseRenderer();
      } else if (renderEnabled)
        windowClosed = true;
    #endif

    // -----------------------------------------------------------------------
    // Cleanup (inside restart loop - freed before next iteration)
    // -----------------------------------------------------------------------
    logger.close();

    for (int i = 0; i < NUM_TIMERS; i++) {
      CUDA_CHECK(cudaEventDestroy(evStart[i]));
      CUDA_CHECK(cudaEventDestroy(evEnd[i]));
    }

    CUDA_CHECK(cudaStreamDestroy(simStream));
    CUDA_CHECK(cudaStreamDestroy(gridStream));

    CUDA_CHECK(cudaEventDestroy(evFork));
    CUDA_CHECK(cudaEventDestroy(evGridDone));
    CUDA_CHECK(cudaEventDestroy(evCaptureReady));

    cudaFree(d_captureBuf);
    cudaFreeHost(h_captureBuf);
    cudaFreeHost(h_snowpackPosX);

    if (params.traceInterval > 0) {
      if (traceStream) CUDA_CHECK(cudaStreamDestroy(traceStream));
      cudaFreeHost(h_posX);    cudaFreeHost(h_posY);    cudaFreeHost(h_posZ);
      cudaFreeHost(h_velX);    cudaFreeHost(h_velY);    cudaFreeHost(h_velZ);
      cudaFreeHost(h_forceX);  cudaFreeHost(h_forceY);  cudaFreeHost(h_forceZ);
      cudaFreeHost(h_attachX); cudaFreeHost(h_attachY); cudaFreeHost(h_attachZ);
      cudaFreeHost(h_mass);
      cudaFreeHost(h_radius);
      cudaFreeHost(h_state);
      cudaFreeHost(h_wetness);
    }

    freeGrid(grid);
    freeParticleSystem(shell);
    freeSnowpack(snowpack);

    isFirstRun = false;
  }
  while (doRestart && renderEnabled && !windowClosed);

  #ifdef ENABLE_VULKAN
    if (renderEnabled) destroyRenderer();
  #endif

  CUDA_CHECK(cudaDeviceReset());

  if (!renderEnabled) {
    auto progEnd = std::chrono::high_resolution_clock::now();
    double progMs = std::chrono::duration<double, std::milli>(progEnd - progStart).count();
    printf("Total program time: %.3f s\n", progMs / 1000.0);
  }

  return 0;
}