#ifndef VULKAN_RENDERER_H
  #define VULKAN_RENDERER_H

  // ============================================================================
  // Vulkan Renderer with CUDA<->Vulkan zero-copy interop
  //
  // Completely separate module: the simulation code does not depend on this.
  // Compiled only when -DENABLE_VULKAN is passed (see Makefile target 'render').
  //
  // Target platform: Jetson Nano (JetPack 4.6+, SM 5.3, Vulkan 1.1)
  // Development:     WSL2 (compile-only, no runtime Vulkan GPU available)
  // ============================================================================

  #ifdef ENABLE_VULKAN
    #include "../../../shared/include/types.h"

    // ============================================================================
    // Configuration for the renderer
    // ============================================================================
    struct RenderConfig {
      int windowWidth = 1280;
      int windowHeight = 720;
      float pointSize = 8.0f;  // base particle point size (pixels)
      float nearPlane = 0.01f;
      float farPlane = 600.0f;
      float fovDeg = 73.5f;
      
      // Camera defaults
      float camTheta = 2.38f;   
      float camPhi = 0.13f;    
      float camDist = 13.5f;  
      float centerX = 6.25f;  
      float centerY = 13.73f;   
      float centerZ = 0.46f;    
      bool trackBall = true;  // fixed camera (press T to toggle tracking)
      bool showBall = false;   // show/hide snowball point sprite
    };

    // ============================================================================
    // Timing info returned each frame
    // ============================================================================
    struct RenderTiming {
      float vboFillMs; // CUDA kernel fillVBO time
      float drawMs;    // Vulkan render + present time
    };

    // ============================================================================
    // Public API - 5 functions, fully opaque internal state
    // ============================================================================

    /**
     * @brief Initialize Vulkan, GLFW window, graphics pipeline, and CUDA<->Vulkan interop.
     * @param cfg     Renderer configuration (window size, camera, point size).
     * @param N       Maximum number of particles to render.
     * @param params  SimParams providing slope geometry.
     * @return 0 on success, non-zero on failure.
     */
    int initRenderer(const RenderConfig &cfg, int N, const SimParams &params);

    /**
     * @brief Fill the shared VBO from particle data and render one frame.
     * @param ps      ParticleSystem providing position and state arrays.
     * @param ball    Current snowball state (rendered as a large point).
     * @param N       Number of particles to render.
     * @param timing  Output: VBO fill and draw timing in milliseconds.
     */
    void renderFrame(const ParticleSystem &ps, const SnowballState &ball, int N, RenderTiming &timing);

    /**
     * @brief Check whether the renderer window should close.
     * @return true if the user pressed X or ESC, false otherwise.
     */
    bool shouldCloseRenderer(void);

    /**
     * @brief Update window title with simulation stats.
     * @param frame          Current frame number.
     * @param shellCount     Total shell particles (seed + captured).
      * @param capturedParticleCount  Cumulative number of particles captured from snowpack.
     * @param simMs          Simulation time in milliseconds.
     * @param vboMs          VBO fill time in milliseconds.
     * @param drawMs         Vulkan draw time in milliseconds.
     */
        void setRendererTitle(int frame, int shellCount, int capturedParticleCount, float simMs, float vboMs, float drawMs);

    /**
     * @brief Set window title to an arbitrary string.
     * @param title  Null-terminated title string.
     */
    void setRendererTitle(const char *title);

    /**
     * @brief Destroy existing VBO interop and recreate with a new capacity.
     * @param newN  New maximum number of particles (VBO will hold newN+1 vertices).
     */
    void resizeRendererVBO(int newN);

    /**
     * @brief Destroy all Vulkan + CUDA interop resources and close the GLFW window.
     */
    void destroyRenderer(void);

    /**
     * @brief Notify the renderer that the simulation has finished.
     *
     * Clears the simRunning flag so the parameter panel becomes interactive
     * again during the post-simulation idle loop.
     */
    void notifySimComplete(void);

    /**
     * @brief Notify the renderer that a new simulation run is starting.
     *
     * Sets simRunning = true and stores the current params for read-only
     * display during the sim loop.  Must be called by main.cu on every
     * restart iteration (first run is handled by runPreSimUI).
     *
     * @param params  The SimParams for this run.
     */
    void notifySimStarted(const SimParams &params);

    /**
     * @brief Check whether the user pressed START in the post-sim idle panel.
     *
     * Single-consumer: returns true once and resets the flag.
     *
     * @return true if START was pressed since the last call.
     */
    bool consumeStartPressed(void);

    /**
     * @brief Copy the renderer's current SimParams (as edited in the idle panel) into @p out.
     *
     * @param out  Destination SimParams.
     */
    void getRendererSimParams(SimParams &out);

    /**
     * @brief Run a pre-simulation UI loop showing editable parameters.
     *
     * Opens the Vulkan window with a Dear ImGui overlay displaying all
     * SimParams fields as editable sliders/inputs plus a "Start Simulation"
     * button.  Blocks until the user presses "Start" or closes the window.
     *
     * @param params  SimParams to display and edit (modified in place).
     * @return true if user pressed "Start Simulation", false if window closed.
     */
    bool runPreSimUI(SimParams &params);
  #endif // ENABLE_VULKAN
#endif // VULKAN_RENDERER_H