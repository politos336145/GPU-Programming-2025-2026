// ============================================================================
// Complete Vulkan setup, CUDA<->Vulkan interop, fillVBO
// kernel, render loop, orbit camera.  Self-contained - no changes needed
// in the simulation code.
//
// Target: Jetson Nano (Vulkan 1.1, SM 5.3)
// Dev:    WSL2 compile-only (libvulkan-dev headers, no runtime GPU)
// ============================================================================

#ifdef ENABLE_VULKAN
  #ifdef _WIN32
    #define VK_USE_PLATFORM_WIN32_KHR
    #ifndef NOMINMAX
      #define NOMINMAX
    #endif
    #include <windows.h>
  #endif

  #define GLFW_INCLUDE_VULKAN

  #include <GLFW/glfw3.h>
  #include <vulkan/vulkan.h>

  #ifdef _WIN32
    #include <vulkan/vulkan_win32.h>
  #endif

  #include <cuda_runtime.h>

  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
  #include <cmath>
  #include <vector>
  #include <algorithm>
  #include <fstream>
  
  #ifndef _WIN32
    #include <unistd.h>
  #endif

  #include "vulkan_renderer.h"
  #include "stb_image.h"

  #include "include/helpers.cuh"
  
  // Dear ImGui (compiled separately in imgui_impl_all.cpp)
  #include "imgui/imgui.h"
  #include "imgui/backends/imgui_impl_glfw.h"
  #include "imgui/backends/imgui_impl_vulkan.h"

  static const int MAX_FRAMES_IN_FLIGHT = 2;

  // ============================================================================
  // ParticleVertex - packed 16 bytes, written by CUDA, read by Vulkan
  // ============================================================================
  struct ParticleVertex {
    float x, y, z;       // world position
    float colorCode;     // 0=seed(green), 2.0–2.99=captured(cyan, wetness-tinted) 3=slope, 4=ground, 5+=snowball
  };

  // ============================================================================
  // SlopeVertex - 20 bytes, textured slope surface (position + UV)
  // ============================================================================
  struct SlopeVertex {
    float x, y, z;       // world position
    float u, v;          // texture coordinates
  };

  // ============================================================================
  // Internal state (file-scope, hidden from caller)
  // ============================================================================
  static struct {
    // GLFW
    GLFWwindow* window = nullptr;

    // Vulkan core
    VkInstance               instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice = VK_NULL_HANDLE;
    VkDevice                 device         = VK_NULL_HANDLE;
    VkQueue                  graphicsQueue  = VK_NULL_HANDLE;
    VkQueue                  presentQueue   = VK_NULL_HANDLE;
    uint32_t                 graphicsFamily = 0;
    uint32_t                 presentFamily  = 0;

    // Swapchain
    VkSwapchainKHR           swapchain      = VK_NULL_HANDLE;
    VkFormat                 swapchainFormat;
    VkExtent2D               swapchainExtent;
    std::vector<VkImage>     swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // Depth
    VkImage                  depthImage      = VK_NULL_HANDLE;
    VkDeviceMemory           depthMemory     = VK_NULL_HANDLE;
    VkImageView              depthImageView  = VK_NULL_HANDLE;

    // Render pass
    VkRenderPass             renderPass     = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuf;

    // Pipeline for particles
    VkPipelineLayout         pipelineLayout    = VK_NULL_HANDLE;
    VkPipeline               particlePipeline  = VK_NULL_HANDLE;

    // Command pool + buffers
    VkCommandPool            commandPool       = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // Sync
    std::vector<VkSemaphore> imageAvailable;
    std::vector<VkSemaphore> renderFinished;
    std::vector<VkFence>     inFlight;
    uint32_t                 currentFrame = 0;

    // Shared VBO (Vulkan buffer + CUDA mapped pointer)
    VkBuffer                 particleBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory           particleBufferMemory = VK_NULL_HANDLE;
    float*                   d_vboPtr             = nullptr;   // CUDA device pointer
    cudaExternalMemory_t     cudaExtMem           = nullptr;
    int                      numParticles         = 0;

    // Push-constant data: MVP matrix (16 floats) + pointSize (1 float)
    // Total 68 bytes ≤ 128 guaranteed minimum
    float mvp[16];
    float pointSize;

    // Camera state
    float camTheta, camPhi, camDist;
    float centerX, centerY, centerZ;
    float fovDeg, nearPlane, farPlane;
    int   winW, winH;

    // Mouse state
    bool  mouseDown  = false;
    double lastMouseX = 0, lastMouseY = 0;

    // Slope geometry
    float slopeSin, slopeCos, slopeHeight;

    // Slope/floor geometry buffer (CPU-managed, separate from CUDA interop)
    VkBuffer       slopeBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory slopeBufferMemory = VK_NULL_HANDLE;
    int            slopeVertexCount  = 0;

    // Triangle pipeline (reuses same shaders, different topology)
    VkPipeline     triPipeline       = VK_NULL_HANDLE;

    // Background sky pipeline (fullscreen procedural night sky)
    VkPipeline       bgPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout bgPipelineLayout = VK_NULL_HANDLE;

    // Slope snow texture
    VkImage          slopeTexImage    = VK_NULL_HANDLE;
    VkDeviceMemory   slopeTexMemory   = VK_NULL_HANDLE;
    VkImageView      slopeTexView     = VK_NULL_HANDLE;
    VkSampler        slopeTexSampler  = VK_NULL_HANDLE;

    // Slope descriptor set (texture sampler)
    VkDescriptorSetLayout slopeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      slopeDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       slopeDescSet    = VK_NULL_HANDLE;

    // Slope textured pipeline (separate shaders + descriptor set)
    VkPipelineLayout slopePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       slopePipeline       = VK_NULL_HANDLE;

    // Slope textured surface buffer (SlopeVertex with UV coordinates)
    VkBuffer       slopeSurfBuf          = VK_NULL_HANDLE;
    VkDeviceMemory slopeSurfBufMemory    = VK_NULL_HANDLE;
    int            slopeSurfVertexCount  = 0;

    // Camera tracking
    bool  trackBall = true;
    bool  showBall  = false;  // show/hide snowball point sprite

    // Timing
    cudaEvent_t evVBOStart, evVBOEnd;

    // VBO fill stream - avoids default-stream sync pitfalls with non-blocking sim streams
    cudaStream_t renderStream = nullptr;

    // Dear ImGui
    bool imguiInitialized = false;

    // Simulation parameters (copy stored at init - shown read-only during sim)
    SimParams simParams;

    // True while the simulation loop is running; false in pre-sim and post-sim
    // idle phases. Controls whether the param panel is disabled or interactive.
    bool simRunning = false;

    // Set to true when the user presses START in the post-sim idle panel.
    // Consumed (and cleared) by consumeStartPressed() in main.cu.
    bool pendingRestart = false;
  } g_ctx;

  // ============================================================================
// Billboard state - defined here so it's visible to the whole TU
// ============================================================================
struct BillboardVertex {
    float x, y, z;
    float u, v;
};

struct BillboardPC {
    float mvp[16];
    float chromaR, chromaG, chromaB;
    float chromaThresh;
};

static struct BillboardState {
    VkImage        img[3]  = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory mem[3]  = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView    view[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkSampler      sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool   = VK_NULL_HANDLE;
    VkDescriptorSet       ds[3]      = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout  = VK_NULL_HANDLE;
    VkPipeline       pipeline        = VK_NULL_HANDLE;
    VkBuffer       vbo    = VK_NULL_HANDLE;
    VkDeviceMemory vboMem = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    float santaPos[3] = {};
    float villPos[3]  = {};
    float santaW = 1.0f,  santaH = 2.0f; // TODO Size
    float villW  = 10.0f, villH  = 5.0f;
    float ck[2][4] = {
        {0.45f, 0.65f, 0.75f, 0.38f},
        {0.06f, 0.16f, 0.33f, 0.22f},
    };
    bool  destroyed = false;
    float destroyX  = 0.0f;
} g_bb;

  // ============================================================================
  // Forward declarations - internal helpers
  // ============================================================================
  static void createInstance(void);
  static void createSurface(void);
  static void pickPhysicalDevice(void);
  static void createLogicalDevice(void);
  static void createSwapchain(void);
  static void createImageViews(void);
  static void createDepthResources(void);
  static void createRenderPass(void);
  static void createFramebuffers(void);
  static void createPipelineLayout(void);
  static void createParticlePipeline(void);
  static void createCommandPool(void);
  static void allocateCommandBuffers(void);
  static void createSyncObjects(void);
  static void createSharedBuffer(int N);
  static void importBufferToCUDA(int N);
  static void createSlopeGeometry(const SimParams& params);
  static void createPipelineCommon(VkPrimitiveTopology topology, VkPipeline& outPipeline);
  static void createTrianglePipeline(void);
  static void createBackgroundPipeline(void);
  static void createSnowTexture(void);
  static void createSlopeDescriptors(void);
  static void createSlopePipeline(void);
  static void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, int N);
  static void initBillboardSystem(const SimParams& params);
  static void resetBillboardState(const SimParams& params);
  static void updateBillboardVBO(const SnowballState& ball);
  static void drawBillboards(VkCommandBuffer cmd);
  static void destroyBillboardSystem(void);
  static void drawBillboardUI(void);

  // ImGui helpers
  static void initImGui(void);
  static void shutdownImGui(void);
  static bool drawParamsUI(SimParams &params, bool editable, bool showStartButton);
  static void drawCameraSceneUI(void);

  // Math helpers
  static void mat4Perspective(float* m, float fovRad, float aspect, float nearP, float farP);
  static void mat4LookAt(float* m, float eyeX, float eyeY, float eyeZ,
                         float atX, float atY, float atZ,
                         float upX, float upY, float upZ);
  static void mat4Multiply(float* out, const float* a, const float* b);

  // Shader loading
  static std::vector<char> readSPIRV(const char* filename);

  // Vulkan helpers
  static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  static VkShaderModule createShaderModule(const std::vector<char>& code);

  // GLFW callbacks
  static void scrollCallback(GLFWwindow* w, double xoff, double yoff);
  static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
  static void cursorPosCallback(GLFWwindow* w, double xpos, double ypos);
  static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);

  // ============================================================================
  // CUDA kernel: fill VBO from SoA particle data
  // ============================================================================
  /**
   * @brief Fill the Vulkan VBO from SoA particle arrays.
   *
   * All shell particles are capture-generated; color is a cyan gradient
   * based on wetness (2.0 + wetness).
   *
   * @param vbo        Output vertex buffer (interleaved x, y, z, colorCode).
   * @param posX       Particle position X.
   * @param posY       Particle position Y.
   * @param posZ       Particle position Z.
   * @param wetness    Per-particle wetness [0,1] (used for color gradient).
   * @param N          Total number of shell particles to render.
   */
  __global__ void fillVBOKernel(ParticleVertex* vbo,
                                const float* __restrict__ posX, const float* __restrict__ posY, const float* __restrict__ posZ,
                                const float* __restrict__ wetness,
                                int N)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    vbo[i].x = posX[i];
    vbo[i].y = posY[i];
    vbo[i].z = posZ[i];
    
    vbo[i].colorCode = 2.0f + wetness[i];
  }

  // ============================================================================
  // Public API: initRenderer
  // ============================================================================
  /**
   * @brief Initialize Vulkan, GLFW window, graphics pipeline, and CUDA<->Vulkan interop.
   * @param cfg     Renderer configuration (window size, camera, point size).
   * @param N       Maximum number of particles to render.
   * @param params  SimParams providing slope geometry.
   * @return 0 on success, non-zero on failure.
   */
  int initRenderer(const RenderConfig& cfg, int N, const SimParams& params) {
    g_ctx.numParticles = N;
    g_ctx.simParams    = params;  // keep a copy for the read-only overlay during simulation
    g_ctx.winW = cfg.windowWidth;
    g_ctx.winH = cfg.windowHeight;
    g_ctx.pointSize  = cfg.pointSize;
    g_ctx.camTheta   = cfg.camTheta;
    g_ctx.camPhi     = cfg.camPhi;
    g_ctx.camDist    = cfg.camDist;
    g_ctx.centerX    = cfg.centerX;
    g_ctx.centerY    = cfg.centerY;
    g_ctx.centerZ    = cfg.centerZ;
    g_ctx.fovDeg     = cfg.fovDeg;
    g_ctx.nearPlane  = cfg.nearPlane;
    g_ctx.farPlane   = cfg.farPlane;
    g_ctx.slopeSin   = params.slopeSin;
    g_ctx.slopeCos   = params.slopeCos;
    g_ctx.slopeHeight = params.slopeHeight;
    g_ctx.trackBall  = cfg.trackBall;

    // CUDA timing events and dedicated stream for VBO fill
    CUDA_CHECK(cudaEventCreate(&g_ctx.evVBOStart));
    CUDA_CHECK(cudaEventCreate(&g_ctx.evVBOEnd));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_ctx.renderStream, cudaStreamNonBlocking));

    // GLFW init
    if (!glfwInit()) {
      fprintf(stderr, "[Renderer] glfwInit failed\n");
      return -1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Vulkan, no OpenGL
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);

    g_ctx.window = glfwCreateWindow(cfg.windowWidth, cfg.windowHeight, "Angry Santa - Vulkan", nullptr, nullptr);
    if (!g_ctx.window) {
      fprintf(stderr, "[Renderer] glfwCreateWindow failed\n");
      glfwTerminate();
      return -1;
    }

    // TODO GLFW callbacks
    //glfwSetScrollCallback(g_ctx.window,       scrollCallback);
    //glfwSetMouseButtonCallback(g_ctx.window,  mouseButtonCallback);
    //glfwSetCursorPosCallback(g_ctx.window,    cursorPosCallback);
    glfwSetKeyCallback(g_ctx.window,          keyCallback);

    // Vulkan setup (order matters)
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createDepthResources();
    createRenderPass();
    createFramebuffers();
    createPipelineLayout();
    createParticlePipeline();
    createTrianglePipeline();
    createBackgroundPipeline();
    createCommandPool();
    allocateCommandBuffers();
    createSyncObjects();

    // Textured slope: snow texture + descriptor set + pipeline
    createSnowTexture();
    createSlopeDescriptors();
    createSlopePipeline();
    printf("[Renderer] Slope pipeline created\n");   // o simile dopo createSlopePipeline()
    initBillboardSystem(params);

    // CUDA<->Vulkan shared buffer
    createSharedBuffer(N);
    importBufferToCUDA(N);

    // Slope/floor geometry (host-managed buffers)
    createSlopeGeometry(params);

    // Dear ImGui overlay
    initImGui();

    printf("[Renderer] Vulkan initialized - %dx%d, %d particles, zero-copy interop\n\n", cfg.windowWidth, cfg.windowHeight, N);
    
    return 0;
  }

  // ============================================================================
  // Public API: renderFrame
  // ============================================================================
  /**
   * @brief Fill the shared VBO from particle data and render one frame.
   * @param ps      ParticleSystem providing position and state arrays.
   * @param ball    Current snowball state (rendered as a large point).
   * @param N       Number of particles to render.
   * @param timing  Output: VBO fill and draw timing in milliseconds.
   */
  void renderFrame(const ParticleSystem& ps, const SnowballState& ball, int N, RenderTiming& timing) {

    // -----------------------------------------------------------------------
    // VBO race-condition guard
    // There is a SINGLE shared particle VBO (CUDA<->Vulkan interop).
    // With MAX_FRAMES_IN_FLIGHT=2 the OTHER slot's Vulkan submission may
    // still be running on the GPU and reading the VBO when CUDA tries to
    // write the next frame's data → illegal memory access.
    // Waiting for ALL in-flight fences before any CUDA write ensures Vulkan
    // has fully released the buffer before we overwrite it.
    // -----------------------------------------------------------------------
    if (!g_ctx.inFlight.empty()) {
      vkWaitForFences(g_ctx.device,
                      static_cast<uint32_t>(g_ctx.inFlight.size()),
                      g_ctx.inFlight.data(),
                      VK_TRUE,
                      UINT64_MAX);
    }

    // --- 1. Fill VBO via CUDA kernel (particles at [0..N-1], snowball at [N]) ---
    //    Uses dedicated renderStream to avoid implicit sync issues between the
    //    legacy default stream (stream 0) and the non-blocking simStream /
    //    gridStream used for physics. On SM 5.0 with CUDA 11.8 the default
    //    stream can trigger illegal-memory-access when it touches a CUDA<->Vulkan
    //    interop buffer while non-blocking streams are alive.
    CUDA_CHECK(cudaEventRecord(g_ctx.evVBOStart, g_ctx.renderStream));
    if (N > 0) {
      const int threadsPerBlock = 256;
      int blocks  = (N + threadsPerBlock - 1) / threadsPerBlock;
      fillVBOKernel<<<blocks, threadsPerBlock, 0, g_ctx.renderStream>>>(
          reinterpret_cast<ParticleVertex*>(g_ctx.d_vboPtr),
          ps.posX, ps.posY, ps.posZ, ps.wetness,
          N);
      CUDA_CHECK(cudaGetLastError());
    }
    
    // TODO Drawing ball
    // Write snowball as vertex N (colorCode=5 → large red ball)
    // Lift position above the slope surface by ball.radius along slope normal
    // so the ball sits ON the slope instead of being half-buried.
    // Slope normal = (slopeSin, slopeCos, 0)
    {
      ParticleVertex ballVert;
      ballVert.x = ball.posX + ball.radius * g_ctx.slopeSin;
      ballVert.y = ball.posY + ball.radius * g_ctx.slopeCos - 0.5f;
      ballVert.z = ball.posZ;
      ballVert.colorCode = 5.0f + ball.radius * 10.0f;  // snowball: encode radius
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<ParticleVertex*>(g_ctx.d_vboPtr) + N,
          &ballVert, sizeof(ParticleVertex), cudaMemcpyHostToDevice,
          g_ctx.renderStream));
    }
    CUDA_CHECK(cudaEventRecord(g_ctx.evVBOEnd, g_ctx.renderStream));
    CUDA_CHECK(cudaEventSynchronize(g_ctx.evVBOEnd));
    CUDA_CHECK(cudaEventElapsedTime(&timing.vboFillMs, g_ctx.evVBOStart, g_ctx.evVBOEnd));
    // On SM 5.0 (Maxwell), cudaEventSynchronize on a NonBlocking stream does not
    // flush the L2 cache boundary between the CUDA compute path and the Vulkan
    // graphics path. Without this full device sync, subsequent simulation kernels
    // can see stale / incoherent data, manifesting as illegal memory access in
    // neighborCollisionKernel. cudaDeviceSynchronize forces a complete L2 flush.
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 2. Auto-track snowball ---
    if (g_ctx.trackBall) {
        
      // Smoothly follow the ball position (exponential smoothing)
      float alpha = 0.05f;  // blend factor (0 = no follow, 1 = snap)
      g_ctx.centerX += alpha * (ball.posX - g_ctx.centerX);
      g_ctx.centerY += alpha * (ball.posY - g_ctx.centerY);
      g_ctx.centerZ += alpha * (ball.posZ - g_ctx.centerZ);
    }

    // --- 3. Build MVP matrix ---
    float eyeX = g_ctx.centerX + g_ctx.camDist * cosf(g_ctx.camPhi) * sinf(g_ctx.camTheta);
    float eyeY = g_ctx.centerY + g_ctx.camDist * sinf(g_ctx.camPhi);
    float eyeZ = g_ctx.centerZ + g_ctx.camDist * cosf(g_ctx.camPhi) * cosf(g_ctx.camTheta);

    float view[16], proj[16];
    mat4LookAt(view, eyeX, eyeY, eyeZ, g_ctx.centerX, g_ctx.centerY, g_ctx.centerZ, 0.0f, 1.0f, 0.0f);
    float aspect = (float)g_ctx.winW / (float)g_ctx.winH;
    mat4Perspective(proj, g_ctx.fovDeg * 3.14159265f / 180.0f, aspect, g_ctx.nearPlane, g_ctx.farPlane);
    // Vulkan clip-space fix: flip Y, adjust Z range [0,1]
    proj[5] *= -1.0f;

    mat4Multiply(g_ctx.mvp, proj, view);

    // --- ImGui new frame ---
    if (g_ctx.imguiInitialized) {
      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      if (g_ctx.simRunning) {
        // During simulation: panel visible but fully disabled (greyed-out).
        // START button is shown but also greyed so the user can see it.
        ImGui::BeginDisabled(true);
        drawParamsUI(g_ctx.simParams, false, true);
        ImGui::EndDisabled();
      } else {
        // Post-simulation idle phase: panel fully interactive again.
        // Capture START press to signal a restart request.
        if (drawParamsUI(g_ctx.simParams, true, true))
          g_ctx.pendingRestart = true;
      }
      drawCameraSceneUI(); // TODO comment to hide
      ImGui::Render();  // finalize draw data for recordCommandBuffer
    }

    updateBillboardVBO(ball);

    // --- 3. Vulkan draw ---
    auto t0 = glfwGetTime();

    uint32_t fi = g_ctx.currentFrame;
    vkWaitForFences(g_ctx.device, 1, &g_ctx.inFlight[fi], VK_TRUE, UINT64_MAX);
    vkResetFences(g_ctx.device, 1, &g_ctx.inFlight[fi]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(g_ctx.device, g_ctx.swapchain, UINT64_MAX,
                          g_ctx.imageAvailable[fi], VK_NULL_HANDLE, &imageIndex);

    vkResetCommandBuffer(g_ctx.commandBuffers[fi], 0);
    recordCommandBuffer(g_ctx.commandBuffers[fi], imageIndex, N);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = &g_ctx.imageAvailable[fi];
    submitInfo.pWaitDstStageMask    = &waitStage;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &g_ctx.commandBuffers[fi];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &g_ctx.renderFinished[fi];

    vkQueueSubmit(g_ctx.graphicsQueue, 1, &submitInfo, g_ctx.inFlight[fi]);

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &g_ctx.renderFinished[fi];
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &g_ctx.swapchain;
    presentInfo.pImageIndices      = &imageIndex;

    vkQueuePresentKHR(g_ctx.presentQueue, &presentInfo);

    g_ctx.currentFrame = (fi + 1) % MAX_FRAMES_IN_FLIGHT;

    timing.drawMs = (float)((glfwGetTime() - t0) * 1000.0);

    glfwPollEvents();
  }

  // ============================================================================
  // Public API: shouldCloseRenderer
  // ============================================================================
  /**
   * @brief Check whether the renderer window should close.
   * @return true if the user pressed X or ESC, false otherwise.
   */
  bool shouldCloseRenderer(void) { return g_ctx.window && glfwWindowShouldClose(g_ctx.window); }

  // ============================================================================
  // Public API: setRendererTitle
  // ============================================================================
  /**
   * @brief Update window title with simulation stats.
   * @param frame          Current frame number.
   * @param shellCount     Total shell particles (seed + captured).
    * @param capturedParticleCount  Cumulative number of particles captured from snowpack.
   * @param simMs          Simulation time in milliseconds.
   * @param vboMs          VBO fill time in milliseconds.
   * @param drawMs         Vulkan draw time in milliseconds.
   */
    void setRendererTitle(int frame, int shellCount, int capturedParticleCount, float simMs, float vboMs, float drawMs) {
    if (!g_ctx.window) return;

    float totalMs = simMs + vboMs + drawMs;
    float fps = (totalMs > 0.001f) ? 1000.0f / totalMs : 0.0f;
    char title[256];
    snprintf(title, sizeof(title),
            "Angry Santa | Frame %d | Shell: %d Captured: %d | "
            "Sim: %.1fms VBO: %.1fms Draw: %.1fms | %.0f FPS",
          frame, shellCount, capturedParticleCount, simMs, vboMs, drawMs, fps);

    glfwSetWindowTitle(g_ctx.window, title);
  }

  /**
   * @brief Set window title to an arbitrary string.
   * @param title  New window title (UTF-8 string).  Ignored if nullptr or if
   */
  void setRendererTitle(const char *title) {
    if (!g_ctx.window || !title) return;

    glfwSetWindowTitle(g_ctx.window, title);
  }

  // ============================================================================
  // Public API: resizeRendererVBO - tear down and recreate shared VBO interop
  // ============================================================================
  /**
   * @brief Destroy existing VBO interop and recreate with a new capacity.
   * @param newN  New maximum number of particles (VBO will hold newN+1 vertices).
   */
  void resizeRendererVBO(int newN) {
    vkDeviceWaitIdle(g_ctx.device);
    cudaDeviceSynchronize();

    // Destroy old CUDA interop
    if (g_ctx.cudaExtMem) {
      cudaDestroyExternalMemory(g_ctx.cudaExtMem);
      g_ctx.cudaExtMem = nullptr;
      g_ctx.d_vboPtr   = nullptr;
    }

    // Destroy old Vulkan buffer + memory
    if (g_ctx.particleBuffer) {
      vkDestroyBuffer(g_ctx.device, g_ctx.particleBuffer, nullptr);
      g_ctx.particleBuffer = VK_NULL_HANDLE;
    }
    if (g_ctx.particleBufferMemory) {
      vkFreeMemory(g_ctx.device, g_ctx.particleBufferMemory, nullptr);
      g_ctx.particleBufferMemory = VK_NULL_HANDLE;
    }

    g_ctx.numParticles = newN;
    createSharedBuffer(newN);
    importBufferToCUDA(newN);

    printf("[Renderer] VBO resized: %d particles + 1 snowball\n", newN);
  }

  // ============================================================================
  // Public API: notifySimComplete / notifySimStarted / consumeStartPressed
  // ============================================================================
  /**
   * @brief Notify the renderer that the simulation loop has completed.
   */
  void notifySimComplete(void) {
    g_ctx.simRunning   = false;
    g_ctx.pendingRestart = false;  // clear any stale flag from previous run
  }

  /**
   * @brief Notify the renderer that a new simulation run is starting.
   *
   * Marks the parameter panel as read-only (disabled) again and stores a
   * copy of the active SimParams for display during the sim loop.
   * Called by main.cu on every restart iteration.
   *
   * @param params  The SimParams that will be used for this run.
   */
  void notifySimStarted(const SimParams &params) {
    g_ctx.simParams  = params;
    g_ctx.simRunning = true;

    // Re-derive slope trig in renderer so geometry matches physics
    g_ctx.slopeSin    = params.slopeSin;
    g_ctx.slopeCos    = params.slopeCos;
    g_ctx.slopeHeight = params.slopeHeight;

    // Recreate slope/ground geometry and reset billboard destroyed state
    if (g_ctx.device) {
      createSlopeGeometry(params);
      resetBillboardState(params);
    }
  }

  /**
   * @brief Check whether the user pressed START in the post-sim idle panel.
   *
   * Returns true once and resets the flag, so it behaves like a
   * single-consumer event.
   *
   * @return true if START was pressed since the last call.
   */
  bool consumeStartPressed(void) {
    bool v = g_ctx.pendingRestart;
    g_ctx.pendingRestart = false;

    return v;
  }

  /**
   * @brief Copy the renderer's current SimParams (as edited by the user in
   *        the post-sim panel) into the caller's SimParams.
   *
   * @param out  Destination to write the renderer's simParams copy.
   */
  void getRendererSimParams(SimParams &out) { out = g_ctx.simParams; }

  /**
   * @brief Destroy all Vulkan + CUDA interop resources and close the GLFW window.
   */
  void destroyRenderer(void) {
    if (g_ctx.device)
      vkDeviceWaitIdle(g_ctx.device);

    destroyBillboardSystem();

    // ImGui cleanup
    shutdownImGui();

    // CUDA cleanup
    if (g_ctx.renderStream) {
      cudaStreamDestroy(g_ctx.renderStream);
      g_ctx.renderStream = nullptr;
    }
    if (g_ctx.cudaExtMem)
      cudaDestroyExternalMemory(g_ctx.cudaExtMem);
    
    CUDA_CHECK(cudaEventDestroy(g_ctx.evVBOStart));
    CUDA_CHECK(cudaEventDestroy(g_ctx.evVBOEnd));

    // Vulkan cleanup (reverse order)
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (g_ctx.inFlight[i])       vkDestroyFence(g_ctx.device, g_ctx.inFlight[i], nullptr);
      if (g_ctx.renderFinished[i]) vkDestroySemaphore(g_ctx.device, g_ctx.renderFinished[i], nullptr);
      if (g_ctx.imageAvailable[i]) vkDestroySemaphore(g_ctx.device, g_ctx.imageAvailable[i], nullptr);
    }
    if (g_ctx.commandPool)         vkDestroyCommandPool(g_ctx.device, g_ctx.commandPool, nullptr);
    if (g_ctx.particlePipeline)    vkDestroyPipeline(g_ctx.device, g_ctx.particlePipeline, nullptr);
    if (g_ctx.triPipeline)         vkDestroyPipeline(g_ctx.device, g_ctx.triPipeline, nullptr);
    if (g_ctx.bgPipeline)          vkDestroyPipeline(g_ctx.device, g_ctx.bgPipeline, nullptr);
    if (g_ctx.slopePipeline)       vkDestroyPipeline(g_ctx.device, g_ctx.slopePipeline, nullptr);
    if (g_ctx.pipelineLayout)      vkDestroyPipelineLayout(g_ctx.device, g_ctx.pipelineLayout, nullptr);
    if (g_ctx.bgPipelineLayout)    vkDestroyPipelineLayout(g_ctx.device, g_ctx.bgPipelineLayout, nullptr);
    if (g_ctx.slopePipelineLayout) vkDestroyPipelineLayout(g_ctx.device, g_ctx.slopePipelineLayout, nullptr);
    if (g_ctx.slopeDescPool)       vkDestroyDescriptorPool(g_ctx.device, g_ctx.slopeDescPool, nullptr);
    if (g_ctx.slopeDescLayout)     vkDestroyDescriptorSetLayout(g_ctx.device, g_ctx.slopeDescLayout, nullptr);
    if (g_ctx.slopeTexSampler)     vkDestroySampler(g_ctx.device, g_ctx.slopeTexSampler, nullptr);
    if (g_ctx.slopeTexView)        vkDestroyImageView(g_ctx.device, g_ctx.slopeTexView, nullptr);
    if (g_ctx.slopeTexImage)       vkDestroyImage(g_ctx.device, g_ctx.slopeTexImage, nullptr);
    if (g_ctx.slopeTexMemory)      vkFreeMemory(g_ctx.device, g_ctx.slopeTexMemory, nullptr);
    for (auto fb : g_ctx.framebuf) vkDestroyFramebuffer(g_ctx.device, fb, nullptr);
    if (g_ctx.renderPass)          vkDestroyRenderPass(g_ctx.device, g_ctx.renderPass, nullptr);
    if (g_ctx.depthImageView)      vkDestroyImageView(g_ctx.device, g_ctx.depthImageView, nullptr);
    if (g_ctx.depthImage)          vkDestroyImage(g_ctx.device, g_ctx.depthImage, nullptr);
    if (g_ctx.depthMemory)         vkFreeMemory(g_ctx.device, g_ctx.depthMemory, nullptr);
    for (auto iv : g_ctx.swapchainImageViews) vkDestroyImageView(g_ctx.device, iv, nullptr);
    if (g_ctx.swapchain)           vkDestroySwapchainKHR(g_ctx.device, g_ctx.swapchain, nullptr);
    if (g_ctx.particleBuffer)      vkDestroyBuffer(g_ctx.device, g_ctx.particleBuffer, nullptr);
    if (g_ctx.particleBufferMemory) vkFreeMemory(g_ctx.device, g_ctx.particleBufferMemory, nullptr);
    if (g_ctx.slopeBuffer)         vkDestroyBuffer(g_ctx.device, g_ctx.slopeBuffer, nullptr);
    if (g_ctx.slopeBufferMemory)   vkFreeMemory(g_ctx.device, g_ctx.slopeBufferMemory, nullptr);
    if (g_ctx.slopeSurfBuf)        vkDestroyBuffer(g_ctx.device, g_ctx.slopeSurfBuf, nullptr);
    if (g_ctx.slopeSurfBufMemory)  vkFreeMemory(g_ctx.device, g_ctx.slopeSurfBufMemory, nullptr);
    if (g_ctx.device)              vkDestroyDevice(g_ctx.device, nullptr);
    if (g_ctx.surface)             vkDestroySurfaceKHR(g_ctx.instance, g_ctx.surface, nullptr);
    if (g_ctx.instance)            vkDestroyInstance(g_ctx.instance, nullptr);

    if (g_ctx.window) {
      glfwDestroyWindow(g_ctx.window);
      g_ctx.window = nullptr;
    }

    glfwTerminate();
    printf("[Renderer] Destroyed\n");
  }

  // ============================================================================
  // Vulkan instance
  // ============================================================================
  /**
   * @brief Create the Vulkan instance with required extensions and validation layers.
   */
  static void createInstance(void) {
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Angry Santa";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName        = "Custom";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_1;

    // Required extensions: GLFW surface + platform
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    // Add external memory capabilities extension
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount       = 0;  // validation layers: add if needed

    #ifdef DEBUG

      // Enable validation layers in debug builds
      const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };
      createInfo.enabledLayerCount   = 1;
      createInfo.ppEnabledLayerNames = validationLayers;
    #endif

    VkResult res = vkCreateInstance(&createInfo, nullptr, &g_ctx.instance);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] vkCreateInstance failed: %d\n", (int)res);
      exit(EXIT_FAILURE);
    }
  }

  // ============================================================================
  // Surface (via GLFW)
  // ============================================================================
  /**
   * @brief Create a Vulkan surface for the GLFW window.
   */
  static void createSurface(void) {
    VkResult res = glfwCreateWindowSurface(g_ctx.instance, g_ctx.window, nullptr, &g_ctx.surface);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] glfwCreateWindowSurface failed: %d\n", (int)res);
      exit(EXIT_FAILURE);
    }
  }

  // ============================================================================
  // Physical device selection (prefer discrete GPU with external memory)
  // ============================================================================
  /**
   * @brief Pick a Vulkan physical device with graphics, present, and external memory support.
   *        Prefers discrete GPUs if multiple suitable devices are found.
   */
  static void pickPhysicalDevice(void) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &count, nullptr);
    if (count == 0) {
      fprintf(stderr, "[Renderer] No Vulkan physical devices found\n");
      exit(EXIT_FAILURE);
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(g_ctx.instance, &count, devices.data());

    // Find a device with graphics + present + external memory support
    for (auto& pd : devices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(pd, &props);

      // Check queue families
      uint32_t qfCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
      std::vector<VkQueueFamilyProperties> qfProps(qfCount);
      vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfProps.data());

      int gfxIdx = -1, presIdx = -1;
      for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
          gfxIdx = (int)i;
        
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g_ctx.surface, &present);
        if (present)
          presIdx = (int)i;
      }

      if (gfxIdx >= 0 && presIdx >= 0) {
    
        // Prefer discrete GPU
        g_ctx.physicalDevice = pd;
        g_ctx.graphicsFamily = (uint32_t)gfxIdx;
        g_ctx.presentFamily  = (uint32_t)presIdx;
        printf("[Renderer] Using GPU: %s\n", props.deviceName);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
          return;  // perfect, stop looking
      }
    }

    if (g_ctx.physicalDevice == VK_NULL_HANDLE) {
      fprintf(stderr, "[Renderer] No suitable GPU found\n");
      exit(EXIT_FAILURE);
    }
  }

  // ============================================================================
  // Logical device + queues
  // ============================================================================
  /**
   * @brief Create the Vulkan logical device with graphics and present queues, and
   *        enable required extensions for swapchain and external memory.
   */
  static void createLogicalDevice(void) {
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;

    // Deduplicate queue families
    uint32_t uniqueFamilies[] = { g_ctx.graphicsFamily, g_ctx.presentFamily };
    int numUnique = (g_ctx.graphicsFamily == g_ctx.presentFamily) ? 1 : 2;

    for (int i = 0; i < numUnique; i++) {
      VkDeviceQueueCreateInfo qi{};
      qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qi.queueFamilyIndex = uniqueFamilies[i];
      qi.queueCount       = 1;
      qi.pQueuePriorities = &priority;
      queueInfos.push_back(qi);
    }

    // Required device extensions (platform-specific for external memory)
    const char* deviceExts[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      #ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
      #else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      #endif
    };

    VkPhysicalDeviceFeatures features{};
    // Enable large points if available (for particle rendering)
    features.largePoints = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount    = (uint32_t)queueInfos.size();
    createInfo.pQueueCreateInfos       = queueInfos.data();
    createInfo.enabledExtensionCount   = 3;
    createInfo.ppEnabledExtensionNames = deviceExts;
    createInfo.pEnabledFeatures        = &features;

    VkResult res = vkCreateDevice(g_ctx.physicalDevice, &createInfo, nullptr, &g_ctx.device);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] vkCreateDevice failed: %d\n", (int)res);
      exit(EXIT_FAILURE);
    }

    vkGetDeviceQueue(g_ctx.device, g_ctx.graphicsFamily, 0, &g_ctx.graphicsQueue);
    vkGetDeviceQueue(g_ctx.device, g_ctx.presentFamily,  0, &g_ctx.presentQueue);
  }

  // ============================================================================
  // Swapchain
  // ============================================================================
  /**
   * @brief Create the Vulkan swapchain, choosing format, present mode, and extent.
   *        Store the swapchain images for later use.
   */
  static void createSwapchain(void) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_ctx.physicalDevice, g_ctx.surface, &caps);

    // Choose format
    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_ctx.physicalDevice, g_ctx.surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_ctx.physicalDevice, g_ctx.surface, &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats) {
      if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosen = f;
        break;
      }
    }
    g_ctx.swapchainFormat = chosen.format;

    // Choose present mode (prefer mailbox, fallback to FIFO)
    uint32_t pmCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_ctx.physicalDevice, g_ctx.surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> modes(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_ctx.physicalDevice, g_ctx.surface, &pmCount, modes.data());

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes) {
      if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
        presentMode = m;
        break;
      }
    }

    // Extent
    g_ctx.swapchainExtent = caps.currentExtent;
    if (caps.currentExtent.width == UINT32_MAX)
      g_ctx.swapchainExtent = { (uint32_t)g_ctx.winW, (uint32_t)g_ctx.winH };

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount)
      imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = g_ctx.surface;
    sci.minImageCount    = imgCount;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = g_ctx.swapchainExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = presentMode;
    sci.clipped          = VK_TRUE;

    if (g_ctx.graphicsFamily != g_ctx.presentFamily) {
      uint32_t indices[] = { g_ctx.graphicsFamily, g_ctx.presentFamily };
      sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
      sci.queueFamilyIndexCount = 2;
      sci.pQueueFamilyIndices   = indices;
    } else
      sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateSwapchainKHR(g_ctx.device, &sci, nullptr, &g_ctx.swapchain);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] vkCreateSwapchainKHR failed: %d\n", (int)res);
      exit(EXIT_FAILURE);
    }

    vkGetSwapchainImagesKHR(g_ctx.device, g_ctx.swapchain, &imgCount, nullptr);
    g_ctx.swapchainImages.resize(imgCount);
    vkGetSwapchainImagesKHR(g_ctx.device, g_ctx.swapchain, &imgCount, g_ctx.swapchainImages.data());
  }

  // ============================================================================
  // Image views for swapchain
  // ============================================================================
  /**
   * @brief Create image views for each swapchain image, for use as render targets.
   *        Store the image views for later use in framebuffers.
   */
  static void createImageViews(void) {
    g_ctx.swapchainImageViews.resize(g_ctx.swapchainImages.size());
    for (size_t i = 0; i < g_ctx.swapchainImages.size(); i++) {
      VkImageViewCreateInfo ci{};
      ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ci.image    = g_ctx.swapchainImages[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format   = g_ctx.swapchainFormat;
      ci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
      ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.baseMipLevel   = 0;
      ci.subresourceRange.levelCount     = 1;
      ci.subresourceRange.baseArrayLayer = 0;
      ci.subresourceRange.layerCount     = 1;

      vkCreateImageView(g_ctx.device, &ci, nullptr, &g_ctx.swapchainImageViews[i]);
    }
  }

  // ============================================================================
  // Depth buffer resources
  // ============================================================================
  /**
   * @brief Create the depth buffer image, allocate memory, and create an image view.
   *        The depth buffer will be used as a depth-stencil attachment in the render pass.
   */
  static void createDepthResources(void) {
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = depthFormat;
    ici.extent        = { g_ctx.swapchainExtent.width, g_ctx.swapchainExtent.height, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(g_ctx.device, &ici, nullptr, &g_ctx.depthImage);

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(g_ctx.device, g_ctx.depthImage, &memReq);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = memReq.size;
    ai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(g_ctx.device, &ai, nullptr, &g_ctx.depthMemory);
    vkBindImageMemory(g_ctx.device, g_ctx.depthImage, g_ctx.depthMemory, 0);

    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = g_ctx.depthImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = depthFormat;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(g_ctx.device, &vci, nullptr, &g_ctx.depthImageView);
  }

  // ============================================================================
  // Render pass (color + depth)
  // ============================================================================
  /**
   * @brief Create a Vulkan render pass with one subpass that uses a color attachment
   *        (the swapchain image) and a depth attachment (the depth buffer).
   */
  static void createRenderPass(void) {
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = g_ctx.swapchainFormat;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAtt{};
    depthAtt.format         = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = { colorAtt, depthAtt };

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments    = attachments;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;

    vkCreateRenderPass(g_ctx.device, &rpci, nullptr, &g_ctx.renderPass);
  }

  // ============================================================================
  // Framebuffers (one per swapchain image)
  // ============================================================================
  /**
   * @brief Create a framebuffer for each swapchain image, using the corresponding image view
   *        and the shared depth image view.  Store the framebuffers for later use in
   *        the rendering loop.
   */
  static void createFramebuffers(void) {
    g_ctx.framebuf.resize(g_ctx.swapchainImageViews.size());
    for (size_t i = 0; i < g_ctx.swapchainImageViews.size(); i++) {
      VkImageView atts[] = { g_ctx.swapchainImageViews[i], g_ctx.depthImageView };

      VkFramebufferCreateInfo fci{};
      fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fci.renderPass      = g_ctx.renderPass;
      fci.attachmentCount = 2;
      fci.pAttachments    = atts;
      fci.width           = g_ctx.swapchainExtent.width;
      fci.height          = g_ctx.swapchainExtent.height;
      fci.layers          = 1;

      vkCreateFramebuffer(g_ctx.device, &fci, nullptr, &g_ctx.framebuf[i]);
    }
  }

  // ============================================================================
  // Pipeline layout (push constants: MVP 4x4 + pointSize = 68 bytes)
  // ============================================================================
  /**
   * @brief Create a pipeline layout with a single push constant range for the vertex shader.
   *        The push constant range is 68 bytes, which can hold a 4x4 MVP matrix (64 bytes) and a float point size (4 bytes).
   */
  static void createPipelineLayout(void) {
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(float) * 17;  // 16 (MVP) + 1 (pointSize) = 68 bytes

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;

    vkCreatePipelineLayout(g_ctx.device, &plci, nullptr, &g_ctx.pipelineLayout);
  }

  // ============================================================================
  // Shared pipeline builder - creates a graphics pipeline with the given
  // primitive topology.  All other state (shaders, vertex input, depth,
  // blend, rasterization) is identical for particles and slope triangles.
  // ============================================================================
  /**
   * @brief Create a graphics pipeline with the specified primitive topology, using shared state for shaders, vertex input, depth testing, blending, and rasterization.
   *        The pipeline will use the global pipeline layout and render pass.
   */
  static void createPipelineCommon(VkPrimitiveTopology topology, VkPipeline& outPipeline) {

    // Load SPIR-V shaders
    auto vertCode = readSPIRV("shaders/particle_vert.spv");
    auto fragCode = readSPIRV("shaders/particle_frag.spv");

    VkShaderModule vertModule = createShaderModule(vertCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    // Vertex input: ParticleVertex (x, y, z, colorCode) = 16 bytes
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(ParticleVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribs[2]{};
    // location 0: position (vec3)
    attribs[0].binding  = 0;
    attribs[0].location = 0;
    attribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attribs[0].offset   = offsetof(ParticleVertex, x);
    // location 1: colorCode (float)
    attribs[1].binding  = 0;
    attribs[1].location = 1;
    attribs[1].format   = VK_FORMAT_R32_SFLOAT;
    attribs[1].offset   = offsetof(ParticleVertex, colorCode);

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viState.vertexBindingDescriptionCount   = 1;
    viState.pVertexBindingDescriptions      = &binding;
    viState.vertexAttributeDescriptionCount = 2;
    viState.pVertexAttributeDescriptions    = attribs;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = topology;

    VkViewport viewport{};
    viewport.x        = 0;
    viewport.y        = 0;
    viewport.width    = (float)g_ctx.swapchainExtent.width;
    viewport.height   = (float)g_ctx.swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = g_ctx.swapchainExtent;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.pViewports    = &viewport;
    vpState.scissorCount  = 1;
    vpState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterState{};
    rasterState.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode    = VK_CULL_MODE_NONE;
    rasterState.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msState{};
    msState.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsState{};
    dsState.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsState.depthTestEnable  = VK_TRUE;
    dsState.depthWriteEnable = VK_TRUE;
    dsState.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cbState{};
    cbState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbState.attachmentCount = 1;
    cbState.pAttachments    = &cbAtt;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &viState;
    pci.pInputAssemblyState = &iaState;
    pci.pViewportState      = &vpState;
    pci.pRasterizationState = &rasterState;
    pci.pMultisampleState   = &msState;
    pci.pDepthStencilState  = &dsState;
    pci.pColorBlendState    = &cbState;
    pci.layout              = g_ctx.pipelineLayout;
    pci.renderPass          = g_ctx.renderPass;
    pci.subpass             = 0;

    VkResult res = vkCreateGraphicsPipelines(g_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &outPipeline);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] vkCreateGraphicsPipelines failed: %d\n", (int)res);
      exit(EXIT_FAILURE);
    }

    vkDestroyShaderModule(g_ctx.device, fragModule, nullptr);
    vkDestroyShaderModule(g_ctx.device, vertModule, nullptr);
  }

  // ============================================================================
  // Particle graphics pipeline (POINT_LIST topology)
  // ============================================================================
  /**
   * @brief Create the graphics pipeline for rendering particles, using VK_PRIMITIVE_TOPOLOGY_POINT_LIST as the input assembly topology.
   *        The pipeline will use the shared state defined in createPipelineCommon, with the point list topology to render each vertex as a point sprite.  The same shaders will be used for both particles and slope triangles, but the input topology will differ.
   */
  static void createParticlePipeline(void) { createPipelineCommon(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, g_ctx.particlePipeline); }

  // ============================================================================
  // Command pool
  // ============================================================================
  /**
   * @brief Create a command pool for allocating command buffers, using the graphics queue family and allowing reset of individual command buffers.
   */
  static void createCommandPool(void) {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g_ctx.graphicsFamily;
    vkCreateCommandPool(g_ctx.device, &cpci, nullptr, &g_ctx.commandPool);
  }

  // ============================================================================
  // Command buffers
  // ============================================================================
  /**
   * @brief Allocate a command buffer for each frame in flight from the command pool.  The command buffers will be recorded each frame with rendering commands and submitted to the graphics queue.
   */
  static void allocateCommandBuffers(void) {
    g_ctx.commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = g_ctx.commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(g_ctx.device, &ai, g_ctx.commandBuffers.data());
  }

  // ============================================================================
  // Synchronization objects
  // ============================================================================
  /**
   * @brief Create semaphores and fences for synchronizing rendering and presentation.  For each frame in flight, create an "image available" semaphore to wait for the swapchain image to be ready, a "render finished" semaphore to signal when rendering is done, and an "in flight" fence to ensure that the CPU waits for the GPU to finish rendering before reusing command buffers and resources.  The fences are created in the signaled state so that the first frame can be rendered without waiting.
   *         The synchronization objects will be used in the rendering loop to coordinate the acquisition of swapchain images, the submission of command buffers, and the presentation of rendered images.  Proper synchronization is crucial to avoid rendering issues and ensure smooth performance.
   */
  static void createSyncObjects(void) {
    g_ctx.imageAvailable.resize(MAX_FRAMES_IN_FLIGHT);
    g_ctx.renderFinished.resize(MAX_FRAMES_IN_FLIGHT);
    g_ctx.inFlight.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkCreateSemaphore(g_ctx.device, &sci, nullptr, &g_ctx.imageAvailable[i]);
      vkCreateSemaphore(g_ctx.device, &sci, nullptr, &g_ctx.renderFinished[i]);
      vkCreateFence(g_ctx.device, &fci, nullptr, &g_ctx.inFlight[i]);
    }
  }

  // ============================================================================
  // Shared buffer: VkBuffer with external memory export (for CUDA import)
  // ============================================================================
  /**
   * @brief Create a Vulkan buffer with the VK_BUFFER_USAGE_VERTEX_BUFFER_BIT usage flag and the VK_EXTERNAL_MEMORY_BUFFER_CREATE_INFO structure in the pNext chain to allow exporting its memory to CUDA.  Allocate device-local memory for the buffer with the VK_EXPORT_MEMORY_ALLOCATE_INFO structure in the pNext chain to specify that the memory can be exported.  Bind the allocated memory to the buffer.  The buffer will be used to store particle vertex data, and will be updated each frame by CUDA kernels via external memory interop.
   */
  static void createSharedBuffer(int N) {
    
    // N+1: extra vertex at index N for the snowball itself
    VkDeviceSize bufSize = (N + 1) * sizeof(ParticleVertex);

    // --- Create buffer with external memory flag ---
    VkExternalMemoryBufferCreateInfo extBufInfo{};
    extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    #ifdef _WIN32
      extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #else
      extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext       = &extBufInfo;
    bci.size        = bufSize;
    bci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateBuffer(g_ctx.device, &bci, nullptr, &g_ctx.particleBuffer);

    // --- Allocate memory with export flag ---
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(g_ctx.device, g_ctx.particleBuffer, &memReq);

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    #ifdef _WIN32
      exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #else
      exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext           = &exportInfo;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(g_ctx.device, &mai, nullptr, &g_ctx.particleBufferMemory);
    vkBindBufferMemory(g_ctx.device, g_ctx.particleBuffer, g_ctx.particleBufferMemory, 0);

    printf("[Renderer] Shared VBO: %d vertices (%d particles + 1 snowball) x %zu bytes = %.2f MB\n",
          N + 1, N, sizeof(ParticleVertex), bufSize / (1024.0f * 1024.0f));
  }

  // ============================================================================
  // Import Vulkan buffer into CUDA via external memory (Win32 handle or POSIX fd)
  // ============================================================================
  /**
   * @brief Import the Vulkan buffer's memory into CUDA using cudaImportExternalMemory.  First, get the memory requirements of the Vulkan buffer to determine the actual allocation size.  Then, fill in a cudaExternalMemoryHandleDesc structure with the appropriate handle type and handle obtained from Vulkan (using vkGetMemoryWin32HandleKHR on Windows or vkGetMemoryFdKHR on Linux).  Call cudaImportExternalMemory to import the memory into CUDA, obtaining a cudaExternalMemory_t handle.  Finally, map the external memory to a CUDA device pointer using cudaExternalMemoryGetMappedBuffer, and store the device pointer for use in CUDA kernels.  The imported buffer will allow CUDA to write particle vertex data directly into the Vulkan buffer that will be used as a vertex buffer in rendering.
   *        Proper error checking is crucial at each step, especially when obtaining the handle from Vulkan and importing the memory into CUDA, as failures can occur due to incorrect handle types, insufficient permissions, or other issues.  The function should print out the obtained handle (Win32 HANDLE or POSIX fd) and the resulting CUDA device pointer for verification.
   *        Note: The buffer size used for CUDA mapping should match the size of the Vulkan buffer allocation, which may be larger than the requested size due to alignment requirements.  Always use the size from vkGetBufferMemoryRequirements when importing and mapping the memory in CUDA.
   *        After importing, the function should also zero-initialize the buffer using cudaMemset to avoid uninitialized reads in the shaders on the first frame before CUDA writes any data.
   * @param N The number of particles (not including the extra snowball vertex) to determine the buffer size.  The actual buffer size will be (N + 1) * sizeof(ParticleVertex).
   */
  static void importBufferToCUDA(int N) {
    VkDeviceSize bufSize = (N + 1) * sizeof(ParticleVertex);

    // Get actual allocation size from Vulkan
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(g_ctx.device, g_ctx.particleBuffer, &memReq);

    cudaExternalMemoryHandleDesc extMemDesc{};
    extMemDesc.size  = memReq.size;
    extMemDesc.flags = 0;

    #ifdef _WIN32
  
      // --- Windows: Win32 HANDLE ---
      VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
      getHandleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
      getHandleInfo.memory     = g_ctx.particleBufferMemory;
      getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

      auto vkGetMemoryWin32HandleKHR_fn = (PFN_vkGetMemoryWin32HandleKHR) vkGetDeviceProcAddr(g_ctx.device, "vkGetMemoryWin32HandleKHR");
      if (!vkGetMemoryWin32HandleKHR_fn) {
        fprintf(stderr, "[Renderer] Could not load vkGetMemoryWin32HandleKHR\n");
        exit(EXIT_FAILURE);
      }

      HANDLE handle = nullptr;
      vkGetMemoryWin32HandleKHR_fn(g_ctx.device, &getHandleInfo, &handle);
      if (!handle) {
        fprintf(stderr, "[Renderer] vkGetMemoryWin32HandleKHR returned invalid handle\n");
        exit(EXIT_FAILURE);
      }

      extMemDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
      extMemDesc.handle.win32.handle = handle;

      printf("[Renderer] CUDA Vulkan interop: Win32 handle=%p", handle);
    #else
    
      // --- Linux / Jetson: POSIX file descriptor ---
      VkMemoryGetFdInfoKHR getFdInfo{};
      getFdInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
      getFdInfo.memory     = g_ctx.particleBufferMemory;
      getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

      auto vkGetMemoryFdKHR_fn = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(g_ctx.device, "vkGetMemoryFdKHR");
      if (!vkGetMemoryFdKHR_fn) {
        fprintf(stderr, "[Renderer] Could not load vkGetMemoryFdKHR\n");
        exit(EXIT_FAILURE);
      }

      int fd = -1;
      vkGetMemoryFdKHR_fn(g_ctx.device, &getFdInfo, &fd);
      if (fd < 0) {
        fprintf(stderr, "[Renderer] vkGetMemoryFdKHR returned invalid fd\n");
        exit(EXIT_FAILURE);
      }

      extMemDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
      extMemDesc.handle.fd = fd;

      printf("[Renderer] CUDA<->Vulkan interop: fd=%d", fd);
    #endif

    CUDA_CHECK(cudaImportExternalMemory(&g_ctx.cudaExtMem, &extMemDesc));

    // Map to CUDA device pointer
    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = bufSize;
    bufDesc.flags  = 0;

    void* devPtr = nullptr;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, g_ctx.cudaExtMem, &bufDesc));
    g_ctx.d_vboPtr = (float*)devPtr;

    // Zero-initialize VBO to avoid uninitialized reads on the very first frame
    CUDA_CHECK(cudaMemset(devPtr, 0, bufSize));

    printf(", CUDA ptr=%p\n", devPtr);
  }

  // ============================================================================
  // Triangle pipeline (TRIANGLE_LIST topology for slope/floor geometry)
  // ============================================================================
  /**
   * @brief Create the graphics pipeline for rendering slope and floor geometry, using VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST as the input assembly topology.  The pipeline will use the shared state defined in createPipelineCommon, with the triangle list topology to render vertices as triangles.  The same shaders will be used for both particles and slope triangles, but the input topology will differ.  This pipeline will be used to render the static geometry of the slope and floor, while the particle pipeline will be used to render the dynamic snow particles.
   */
  static void createTrianglePipeline(void) {
    createPipelineCommon(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, g_ctx.triPipeline);
    printf("[Renderer] Triangle pipeline created for slope geometry\n");
  }

  // ============================================================================
  // Background sky pipeline - fullscreen procedural night sky
  //
  // Uses a single fullscreen triangle (no vertex input, gl_VertexIndex trick).
  // Separate pipeline layout: no push constants, no descriptors.
  // Depth test ON, depth write OFF - drawn first, covers entire screen at max
  // depth so all subsequent geometry naturally occludes it.
  // ============================================================================
  /**
   * @brief Create a graphics pipeline for rendering the background sky, using a fullscreen triangle with no vertex input.
   *        The vertex shader will generate positions from gl_VertexIndex to cover the entire screen, and the fragment shader will procedurally generate a night sky.
   *        The pipeline layout will be separate from the particle and triangle pipelines, with no push constants or descriptors, since the background shader does not need any dynamic data.
   *        Depth testing will be enabled but depth writing will be disabled, so that the background will be drawn at maximum depth and all subsequent geometry will naturally occlude it without needing to sort draw calls.
   *        This pipeline will be used to render the static background before rendering any particles or slope geometry.
   *        Proper error handling is included in case the pipeline creation fails, with a fallback to using a clear color if the pipeline cannot be created.
   *        The function will print out a message indicating whether the background pipeline was successfully created or if it fell back to the clear color.
   */
  static void createBackgroundPipeline(void) {
    auto vertCode = readSPIRV("shaders/background_vert.spv");
    auto fragCode = readSPIRV("shaders/background_frag.spv");
    VkShaderModule vertModule = createShaderModule(vertCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    // No vertex input - positions generated from gl_VertexIndex
    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width    = (float)g_ctx.swapchainExtent.width;
    viewport.height   = (float)g_ctx.swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = g_ctx.swapchainExtent;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.pViewports    = &viewport;
    vpState.scissorCount  = 1;
    vpState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterState{};
    rasterState.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode    = VK_CULL_MODE_NONE;
    rasterState.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msState{};
    msState.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test ON to participate in sorting, but NO write so geometry overwrites it
    VkPipelineDepthStencilStateCreateInfo dsState{};
    dsState.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsState.depthTestEnable  = VK_TRUE;
    dsState.depthWriteEnable = VK_FALSE;
    dsState.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cbState{};
    cbState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbState.attachmentCount = 1;
    cbState.pAttachments    = &cbAtt;

    // Empty pipeline layout - no push constants, no descriptors
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkCreatePipelineLayout(g_ctx.device, &plci, nullptr, &g_ctx.bgPipelineLayout);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &viState;
    pci.pInputAssemblyState = &iaState;
    pci.pViewportState      = &vpState;
    pci.pRasterizationState = &rasterState;
    pci.pMultisampleState   = &msState;
    pci.pDepthStencilState  = &dsState;
    pci.pColorBlendState    = &cbState;
    pci.layout              = g_ctx.bgPipelineLayout;
    pci.renderPass          = g_ctx.renderPass;
    pci.subpass             = 0;

    VkResult res = vkCreateGraphicsPipelines(g_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &g_ctx.bgPipeline);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] Background pipeline creation failed: %d\n", (int)res);
      // Non-fatal: fall back to clear color
      g_ctx.bgPipeline = VK_NULL_HANDLE;
    }

    vkDestroyShaderModule(g_ctx.device, fragModule, nullptr);
    vkDestroyShaderModule(g_ctx.device, vertModule, nullptr);
    printf("[Renderer] Background sky pipeline created\n");
  }

  // ============================================================================
  // Snow texture - 512×512 RGBA8 uploaded to VkImage
  //
  // Tries to load "snow.png" (or "shaders/snow.png") via stb_image.
  // If the file is not found, falls back to a procedural noise-based snow
  // surface (white + slight blue/grey variations, sparkle highlights).
  // Uses a staging buffer → device-local VkImage copy with layout transitions.
  // ============================================================================
  /**
   * @brief Create a snow texture as a 512×512 RGBA8 VkImage.  First, attempt to load "snow.png" from the current directory or common subdirectories using stb_image.
   *        If the file is found and loaded successfully, use its pixel data for the texture.
   *        If the file is not found, procedurally generate a snow texture using layered smooth noise to create white with slight blue/grey variations and sparkle highlights.
   *        Then, create a staging buffer, copy the pixel data into it, and create a device-local VkImage for the snow texture.
   *        Use vkCmdCopyBufferToImage with appropriate layout transitions to transfer the pixel data from the staging buffer to the VkImage.
   *        Finally, clean up the staging buffer and any loaded pixel data.
   *        The resulting snow texture will be used in the particle shader to render snow particles.
   */
  static void createSnowTexture(void) {
    uint32_t TEX_W = 512, TEX_H = 512;
    uint8_t* pixelData = nullptr;
    bool fromFile = false;

    // ---- 1. Try loading from file ----
    const char* path = EXEPATH "textures/snow.png"; // defined in CMakeLists
    int fileW = 0, fileH = 0, fileChannels = 0;
    pixelData = stbi_load(path, &fileW, &fileH, &fileChannels, 4);
    if (pixelData) {
      TEX_W = (uint32_t)fileW;
      TEX_H = (uint32_t)fileH;
      fromFile = true;
    } else
      printf("[Renderer] Failed to load texture: %s\n", stbi_failure_reason());

    // ---- 2. Fallback: generate procedural snow pixels ----
    std::vector<uint8_t> proceduralPixels;
    if (!fromFile) {
      TEX_W = 512; TEX_H = 512;
      proceduralPixels.resize(TEX_W * TEX_H * 4);

      // Simple integer hash → float in [-1, 1]
      auto hashf = [](int x, int y) -> float {
        int n = x + y * 137;
        n = (n << 13) ^ n;
        return 1.0f - (float)((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
      };

      // Bilinear smooth noise
      auto smoothNoise = [&](float fx, float fy) -> float {
        int   ix = (int)floorf(fx), iy = (int)floorf(fy);
        float fracX = fx - ix, fracY = fy - iy;
        float v00 = hashf(ix,   iy),   v10 = hashf(ix+1, iy);
        float v01 = hashf(ix,   iy+1), v11 = hashf(ix+1, iy+1);
        float i0 = v00 + fracX * (v10 - v00);
        float i1 = v01 + fracX * (v11 - v01);
        return i0 + fracY * (i1 - i0);
      };

      for (uint32_t y = 0; y < TEX_H; y++) {
        for (uint32_t x = 0; x < TEX_W; x++) {
          float nx = (float)x / TEX_W, ny = (float)y / TEX_H;
          float n  = smoothNoise(nx * 8.0f,  ny * 8.0f)  * 0.50f
                   + smoothNoise(nx * 16.0f, ny * 16.0f) * 0.25f
                   + smoothNoise(nx * 32.0f, ny * 32.0f) * 0.125f;
          n = 0.5f + n;
          n = fmaxf(0.0f, fminf(1.0f, n));

          float r = 0.84f + n * 0.16f;
          float g = 0.86f + n * 0.14f;
          float b = 0.91f + n * 0.09f;

          float sparkle = hashf(x * 7, y * 13);
          if (sparkle > 0.96f) { r = 1.0f; g = 1.0f; b = 1.0f; }

          float streak = smoothNoise(nx * 3.0f + 0.5f, ny * 50.0f);
          if (streak > 0.7f) { r *= 0.92f; g *= 0.94f; }

          uint32_t idx = (y * TEX_W + x) * 4;
          proceduralPixels[idx + 0] = (uint8_t)(fminf(r, 1.0f) * 255.0f);
          proceduralPixels[idx + 1] = (uint8_t)(fminf(g, 1.0f) * 255.0f);
          proceduralPixels[idx + 2] = (uint8_t)(fminf(b, 1.0f) * 255.0f);
          proceduralPixels[idx + 3] = 255;
        }
      }
      pixelData = proceduralPixels.data();
      printf("[Renderer] Using procedural snow texture (%ux%u)\n", TEX_W, TEX_H);
    }

    const VkDeviceSize imageSize = TEX_W * TEX_H * 4;

    // ---- 3. Create staging buffer ----
    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = imageSize;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(g_ctx.device, &bci, nullptr, &stagingBuffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(g_ctx.device, stagingBuffer, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(g_ctx.device, &mai, nullptr, &stagingMemory);
    vkBindBufferMemory(g_ctx.device, stagingBuffer, stagingMemory, 0);

    void* data;
    vkMapMemory(g_ctx.device, stagingMemory, 0, imageSize, 0, &data);
    memcpy(data, pixelData, imageSize);
    vkUnmapMemory(g_ctx.device, stagingMemory);

    // Free stb_image memory if loaded from file
    if (fromFile && pixelData) stbi_image_free(pixelData);

    // ---- 3. Create VkImage (device-local, optimal tiling) ----
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent        = {TEX_W, TEX_H, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(g_ctx.device, &ici, nullptr, &g_ctx.slopeTexImage);

    vkGetImageMemoryRequirements(g_ctx.device, g_ctx.slopeTexImage, &memReq);
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(g_ctx.device, &mai, nullptr, &g_ctx.slopeTexMemory);
    vkBindImageMemory(g_ctx.device, g_ctx.slopeTexImage, g_ctx.slopeTexMemory, 0);

    // ---- 4. Transition + copy via single-use command buffer ----
    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = g_ctx.commandPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_ctx.device, &cbai, &cmdBuf);

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &cbbi);

    // Transition: UNDEFINED → TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = g_ctx.slopeTexImage;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask       = 0;
    barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy staging buffer → image
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {TEX_W, TEX_H, 1};
    vkCmdCopyBufferToImage(cmdBuf, stagingBuffer, g_ctx.slopeTexImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition: TRANSFER_DST → SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmdBuf;
    vkQueueSubmit(g_ctx.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_ctx.graphicsQueue);

    vkFreeCommandBuffers(g_ctx.device, g_ctx.commandPool, 1, &cmdBuf);
    vkDestroyBuffer(g_ctx.device, stagingBuffer, nullptr);
    vkFreeMemory(g_ctx.device, stagingMemory, nullptr);

    // ---- 5. Image view ----
    VkImageViewCreateInfo ivci{};
    ivci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image            = g_ctx.slopeTexImage;
    ivci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format           = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(g_ctx.device, &ivci, nullptr, &g_ctx.slopeTexView);

    // ---- 6. Sampler (linear filtering, repeat wrap) ----
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkCreateSampler(g_ctx.device, &sci, nullptr, &g_ctx.slopeTexSampler);

    printf("[Renderer] Snow texture ready (%ux%u, %s)\n", TEX_W, TEX_H,
           fromFile ? "from file" : "procedural");
  }

  // ============================================================================
  // Slope descriptors - descriptor set layout, pool, set for the snow texture
  // ============================================================================
  /**
   * @brief Create the descriptor set layout, descriptor pool, and allocate a descriptor set for the slope texture.
   *        The descriptor set layout will have a single combined image sampler at binding 0 for the snow texture.
   *        The descriptor pool will be created with enough capacity for one combined image sampler.
   *        A descriptor set will be allocated from the pool using the layout, and then updated to point to the snow texture's image view and sampler.
   *        This descriptor set will be bound in the slope pipeline to allow the fragment shader to sample from the snow texture when rendering slope geometry.
   */
  static void createSlopeDescriptors(void) {

    // Layout: 1 combined image sampler at binding 0
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &binding;
    vkCreateDescriptorSetLayout(g_ctx.device, &layoutInfo, nullptr, &g_ctx.slopeDescLayout);

    // Pool: exactly 1 combined image sampler
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = 1;
    vkCreateDescriptorPool(g_ctx.device, &poolInfo, nullptr, &g_ctx.slopeDescPool);

    // Allocate set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = g_ctx.slopeDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &g_ctx.slopeDescLayout;
    vkAllocateDescriptorSets(g_ctx.device, &allocInfo, &g_ctx.slopeDescSet);

    // Update with snow texture + sampler
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView   = g_ctx.slopeTexView;
    imageInfo.sampler     = g_ctx.slopeTexSampler;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = g_ctx.slopeDescSet;
    write.dstBinding      = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo      = &imageInfo;
    vkUpdateDescriptorSets(g_ctx.device, 1, &write, 0, nullptr);

    printf("[Renderer] Slope texture descriptor set ready\n");
  }

  // ============================================================================
  // Slope textured pipeline - uses slope.vert/slope.frag + snow texture
  //
  // Pipeline layout: push constants (MVP 64 B + pointSize 4 B = 68 B shared)
  //                + 1 descriptor set (combined image sampler for snow texture)
  // ============================================================================
  /**
   * @brief Create the graphics pipeline for rendering slope geometry, using the slope.vert and slope.frag shaders, and binding the snow texture via a descriptor set.
   *        The pipeline layout will include a push constant range of 68 bytes (for the MVP matrix and point size) that is shared with the particle pipeline, as well as a descriptor set layout that includes a combined image sampler for the snow texture.
   *        The input assembly topology will be VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST to render the slope geometry as triangles.
   *        This pipeline will be used to render the static slope and floor geometry with texturing from the snow texture.
   */
  static void createSlopePipeline(void) {

    // Pipeline layout: shared push constant range + texture descriptor set
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(float) * 17;  // matches particle push constant range

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &g_ctx.slopeDescLayout;
    vkCreatePipelineLayout(g_ctx.device, &plci, nullptr, &g_ctx.slopePipelineLayout);

    // Load slope shaders
    auto vertCode = readSPIRV("shaders/slope_vert.spv");
    auto fragCode = readSPIRV("shaders/slope_frag.spv");
    VkShaderModule vertModule = createShaderModule(vertCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    // Vertex input: SlopeVertex (x, y, z, u, v) = 20 bytes
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(SlopeVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribs[2]{};
    // location 0: position (vec3)
    attribs[0].binding  = 0;
    attribs[0].location = 0;
    attribs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attribs[0].offset   = offsetof(SlopeVertex, x);
    // location 1: UV (vec2)
    attribs[1].binding  = 0;
    attribs[1].location = 1;
    attribs[1].format   = VK_FORMAT_R32G32_SFLOAT;
    attribs[1].offset   = offsetof(SlopeVertex, u);

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viState.vertexBindingDescriptionCount   = 1;
    viState.pVertexBindingDescriptions      = &binding;
    viState.vertexAttributeDescriptionCount = 2;
    viState.pVertexAttributeDescriptions    = attribs;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width    = (float)g_ctx.swapchainExtent.width;
    viewport.height   = (float)g_ctx.swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = g_ctx.swapchainExtent;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.pViewports    = &viewport;
    vpState.scissorCount  = 1;
    vpState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterState{};
    rasterState.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode    = VK_CULL_MODE_NONE;
    rasterState.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msState{};
    msState.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsState{};
    dsState.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsState.depthTestEnable  = VK_TRUE;
    dsState.depthWriteEnable = VK_TRUE;
    dsState.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt.blendEnable         = VK_TRUE;
    cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbAtt.colorBlendOp        = VK_BLEND_OP_ADD;
    cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cbAtt.alphaBlendOp        = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo cbState{};
    cbState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbState.attachmentCount = 1;
    cbState.pAttachments    = &cbAtt;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &viState;
    pci.pInputAssemblyState = &iaState;
    pci.pViewportState      = &vpState;
    pci.pRasterizationState = &rasterState;
    pci.pMultisampleState   = &msState;
    pci.pDepthStencilState  = &dsState;
    pci.pColorBlendState    = &cbState;
    pci.layout              = g_ctx.slopePipelineLayout;
    pci.renderPass          = g_ctx.renderPass;
    pci.subpass             = 0;

    VkResult res = vkCreateGraphicsPipelines(g_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr, &g_ctx.slopePipeline);
    if (res != VK_SUCCESS) {
      fprintf(stderr, "[Renderer] Slope texture pipeline creation failed: %d\n", (int)res);
      g_ctx.slopePipeline = VK_NULL_HANDLE;
    }

    vkDestroyShaderModule(g_ctx.device, fragModule, nullptr);
    vkDestroyShaderModule(g_ctx.device, vertModule, nullptr);
    printf("[Renderer] Slope textured pipeline created\n");
  }

  // ============================================================================
  // Create slope/floor geometry - host-managed VBOs (not CUDA interop)
  //
  // Two separate buffers:
  //   1. slopeSurfBuffer (SlopeVertex): textured slope surface with UV coords
  //   2. slopeBuffer     (ParticleVertex): flat ground at Y=0 (colorCode=4, brown)
  // ============================================================================
  /**
   * @brief Create the vertex buffers for the slope and ground geometry. 
   *        The slope surface will be represented by an array of SlopeVertex structures, which include position and UV coordinates for texturing with the snow texture.
   *        The ground plane will be represented by an array of ParticleVertex structures, which will use a flat color (colorCode=4) to render a brown floor. 
   *        Both buffers will be created as host-visible and host-coherent, allowing the CPU to write vertex data directly into them without needing staging buffers or CUDA interop.
   *        The slope geometry will consist of two triangles forming a rectangle that extends from the top of the slope to the ground, while the ground geometry will consist of two triangles forming a rectangle that extends from where the slope meets the ground to some distance along the X-axis.
   *        The resulting vertex buffers will be bound in the slope pipeline to render the static slope
   */
  static void createSlopeGeometry(const SimParams& params) {
    float sn = g_ctx.slopeSin;
    float cs = g_ctx.slopeCos;
    float H  = g_ctx.slopeHeight;

    // Free previous buffers if they exist (called on restart)
    if (g_ctx.slopeSurfBuf) {
      vkDestroyBuffer(g_ctx.device, g_ctx.slopeSurfBuf, nullptr);
      g_ctx.slopeSurfBuf = VK_NULL_HANDLE;
    }
    if (g_ctx.slopeSurfBufMemory) {
      vkFreeMemory(g_ctx.device, g_ctx.slopeSurfBufMemory, nullptr);
      g_ctx.slopeSurfBufMemory = VK_NULL_HANDLE;
    }
    if (g_ctx.slopeBuffer) {
      vkDestroyBuffer(g_ctx.device, g_ctx.slopeBuffer, nullptr);
      g_ctx.slopeBuffer = VK_NULL_HANDLE;
    }
    if (g_ctx.slopeBufferMemory) {
      vkFreeMemory(g_ctx.device, g_ctx.slopeBufferMemory, nullptr);
      g_ctx.slopeBufferMemory = VK_NULL_HANDLE;
    }

    // Slope top: s=0 → world (x=0, y=H/cs)
    // Slope bottom: clamped to where the slope meets y=0 (groundStartX)
    // Uses same basis change as initSnowpackKernel: worldX = s*cs, worldY = H/cs - s*sn
    float slopeTopX = 0.0f;
    float slopeTopY = H / cs;
    float slopeEndS = params.spawnStartS + params.spawnLengthS + 5.0f; // +5 m visual margin
    float slopeBotX = slopeEndS * cs;
    float slopeBotY = (H / cs) - slopeEndS * sn;

    // Clamp slope bottom to y=0: the slope should not extend below the ground plane.
    // Where slope meets y=0: s_ground = H/(cs*sn+sn*...) → x = H/sn, y = 0
    float groundStartX = H / sn;
    if (slopeBotY < 0.0f) {
      slopeBotX = groundStartX;
      slopeBotY = 0.0f;
    }

    // Lateral half-width from spawnWidthZ + 2 m visual margin
    float lateralHalf = params.spawnWidthZ * 0.5f + 2.0f;

    // ---- Textured slope surface (SlopeVertex with UV) ----
    // Compute tiling from world dimensions so texture isn't stretched
    float slopeLen = sqrtf((slopeBotX - slopeTopX) * (slopeBotX - slopeTopX) +
                           (slopeTopY - slopeBotY) * (slopeTopY - slopeBotY));
    float tileSize = 5.0f;  // one texture tile = 5 world units
    float tilingU  = (lateralHalf * 2.0f) / tileSize;
    float tilingV  = slopeLen / tileSize;

    SlopeVertex slopeSurf[6];
    slopeSurf[0] = {slopeTopX, slopeTopY, -lateralHalf, 0.0f,    0.0f};
    slopeSurf[1] = {slopeBotX, slopeBotY, -lateralHalf, 0.0f,    tilingV};
    slopeSurf[2] = {slopeTopX, slopeTopY,  lateralHalf, tilingU,  0.0f};
    slopeSurf[3] = {slopeBotX, slopeBotY, -lateralHalf, 0.0f,    tilingV};
    slopeSurf[4] = {slopeBotX, slopeBotY,  lateralHalf, tilingU,  tilingV};
    slopeSurf[5] = {slopeTopX, slopeTopY,  lateralHalf, tilingU,  0.0f};
    g_ctx.slopeSurfVertexCount = 6;

    // Upload textured slope to slopeSurfBuffer
    {
      VkDeviceSize bufSize = sizeof(slopeSurf);
      VkBufferCreateInfo bci{};
      bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size        = bufSize;
      bci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      vkCreateBuffer(g_ctx.device, &bci, nullptr, &g_ctx.slopeSurfBuf);

      VkMemoryRequirements memReq;
      vkGetBufferMemoryRequirements(g_ctx.device, g_ctx.slopeSurfBuf, &memReq);

      VkMemoryAllocateInfo mai{};
      mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize  = memReq.size;
      mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      vkAllocateMemory(g_ctx.device, &mai, nullptr, &g_ctx.slopeSurfBufMemory);
      vkBindBufferMemory(g_ctx.device, g_ctx.slopeSurfBuf, g_ctx.slopeSurfBufMemory, 0);

      void* data;
      vkMapMemory(g_ctx.device, g_ctx.slopeSurfBufMemory, 0, bufSize, 0, &data);
      memcpy(data, slopeSurf, bufSize);
      vkUnmapMemory(g_ctx.device, g_ctx.slopeSurfBufMemory);
    }

    // ---- Ground plane (ParticleVertex, flat color via colorCode=4) ----
    // Extends from where slope meets y=0 (groundStartX) onwards
    float groundColor = 4.0f;
    float gndEndX = groundStartX + 20.0f;
    ParticleVertex groundVerts[6];
    groundVerts[0] = {groundStartX, 0.0f, -lateralHalf, groundColor};
    groundVerts[1] = {gndEndX,      0.0f, -lateralHalf, groundColor};
    groundVerts[2] = {groundStartX, 0.0f,  lateralHalf, groundColor};
    groundVerts[3] = {gndEndX,      0.0f, -lateralHalf, groundColor};
    groundVerts[4] = {gndEndX,      0.0f,  lateralHalf, groundColor};
    groundVerts[5] = {groundStartX, 0.0f,  lateralHalf, groundColor};
    g_ctx.slopeVertexCount = 6;

    // Upload ground to slopeBuffer
    {
      VkDeviceSize bufSize = sizeof(groundVerts);
      VkBufferCreateInfo bci{};
      bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size        = bufSize;
      bci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      vkCreateBuffer(g_ctx.device, &bci, nullptr, &g_ctx.slopeBuffer);

      VkMemoryRequirements memReq;
      vkGetBufferMemoryRequirements(g_ctx.device, g_ctx.slopeBuffer, &memReq);

      VkMemoryAllocateInfo mai{};
      mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize  = memReq.size;
      mai.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      vkAllocateMemory(g_ctx.device, &mai, nullptr, &g_ctx.slopeBufferMemory);
      vkBindBufferMemory(g_ctx.device, g_ctx.slopeBuffer, g_ctx.slopeBufferMemory, 0);

      void* data;
      vkMapMemory(g_ctx.device, g_ctx.slopeBufferMemory, 0, bufSize, 0, &data);
      memcpy(data, groundVerts, bufSize);
      vkUnmapMemory(g_ctx.device, g_ctx.slopeBufferMemory);
    }

    printf("[Renderer] Slope geometry: %d textured slope + %d ground vertices\n", g_ctx.slopeSurfVertexCount, g_ctx.slopeVertexCount);
  }

  // ============================================================================
  // Record command buffer for one frame
  // ============================================================================
  /**
   * @brief Record the Vulkan command buffer for rendering one frame. This includes:
   *        1. Binding the background pipeline and drawing a fullscreen triangle for the sky.
   *        2. Binding the slope pipeline, pushing constants for MVP and point size, binding the slope descriptor set, and drawing the slope surface geometry.
   *        3. Binding the triangle pipeline, pushing constants, and drawing the ground plane geometry.
   *        4. Binding the particle pipeline, pushing constants, and drawing the particles and snowball as points.
   *        5. Rendering the Dear ImGui overlay if initialized.
   * @param cmd The command buffer to record into.
   * @param imageIndex The index of the swapchain image being rendered to, used for framebuffer binding.
   * @param N The number of particles to draw (the snowball is drawn as the N+1-th point).  
   */
  static void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, int N) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Clear values: dark blue-grey background + depth
    VkClearValue clearValues[2]{};
    clearValues[0].color        = {{0.05f, 0.05f, 0.12f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = g_ctx.renderPass;
    rpBegin.framebuffer       = g_ctx.framebuf[imageIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = g_ctx.swapchainExtent;
    rpBegin.clearValueCount   = 2;
    rpBegin.pClearValues      = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // --- Draw 0: Background sky (fullscreen procedural triangle) ---
    if (g_ctx.bgPipeline) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.bgPipeline);
      vkCmdDraw(cmd, 3, 1, 0, 0);  // fullscreen triangle, no VBO
    }

    // Push constants: MVP (16 floats) + pointSize (1 float)
    float pushData[17];
    memcpy(pushData, g_ctx.mvp, 16 * sizeof(float));
    pushData[16] = g_ctx.pointSize;

    // --- Draw 1: Textured slope surface (snow texture pipeline) ---
    if (g_ctx.slopePipeline && g_ctx.slopeSurfBuf && g_ctx.slopeSurfVertexCount > 0) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.slopePipeline);
      pushData[16] = 1.0f;  // pointSize unused for triangles, but must be valid
      vkCmdPushConstants(cmd, g_ctx.slopePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.slopePipelineLayout,
                              0, 1, &g_ctx.slopeDescSet, 0, nullptr);
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &g_ctx.slopeSurfBuf, &offset);
      vkCmdDraw(cmd, (uint32_t)g_ctx.slopeSurfVertexCount, 1, 0, 0);
    }

    // --- Draw 2: Ground plane (flat-color triangle pipeline) ---
    if (g_ctx.triPipeline && g_ctx.slopeBuffer && g_ctx.slopeVertexCount > 0) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.triPipeline);
      pushData[16] = 1.0f;
      vkCmdPushConstants(cmd, g_ctx.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &g_ctx.slopeBuffer, &offset);
      vkCmdDraw(cmd, (uint32_t)g_ctx.slopeVertexCount, 1, 0, 0);
    }

    // --- Draw 3: Particles + snowball (N particles + 1 ball = N+1 points) ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.particlePipeline);
    pushData[16] = g_ctx.pointSize;
    vkCmdPushConstants(cmd, g_ctx.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_ctx.particleBuffer, &offset);
    uint32_t drawCount = g_ctx.showBall ? (uint32_t)(N + 1) : (uint32_t)N;
    vkCmdDraw(cmd, drawCount, 1, 0, 0); // N shell particles [+ 1 snowball]

    // --- Draw 4: Billboard sprites (Santa + Village) ---
    drawBillboards(cmd);
    
    // --- Draw 5: Dear ImGui overlay ---
    if (g_ctx.imguiInitialized) {
      ImDrawData* drawData = ImGui::GetDrawData();
      if (drawData)
        ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  // ============================================================================
  // Shader loading: read SPIR-V binary from file
  // ============================================================================
  /**
   * @brief Read a SPIR-V shader binary from a file and return its contents as a vector of bytes. The function first attempts to open the file using the provided filename, and if that fails, it tries to construct an alternative path relative to the executable's directory (to handle cases where the current working directory is different from the build directory). If the file cannot be opened from either location, an error message is printed and the program exits. If successful, the file size is determined, a buffer of the appropriate size is allocated, and the file contents are read into the buffer, which is then returned.
   * @param filename The name of the SPIR-V binary file to read.
   * @return A vector of bytes containing the SPIR-V shader binary.
   */
  static std::vector<char> readSPIRV(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    // Fallback: try relative to executable directory (handles CWD ≠ build/)
    std::string altPath;
    if (!file.is_open()) {
      #ifdef _WIN32
        char exePath[MAX_PATH];
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        altPath = std::string(exePath);
        auto pos = altPath.find_last_of("\\/");
        if (pos != std::string::npos)
          altPath = altPath.substr(0, pos + 1);
        
        altPath += filename;
      #else
        // Linux: read /proc/self/exe
        char exePath[4096];
        ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
        if (len > 0) {
          exePath[len] = '\0';
          altPath = std::string(exePath);
          auto pos = altPath.find_last_of('/');
          if (pos != std::string::npos)
            altPath = altPath.substr(0, pos + 1);
          
          altPath += filename;
        }
      #endif
      if (!altPath.empty())
        file.open(altPath, std::ios::ate | std::ios::binary);
    }

    if (!file.is_open()) {
      fprintf(stderr, "[Renderer] Cannot open shader: %s\n", filename);
      if (!altPath.empty())
        fprintf(stderr, "[Renderer]   also tried: %s\n", altPath.c_str());
      
      exit(EXIT_FAILURE);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), (std::streamsize)fileSize);
    
    return buffer;
  }

  // ============================================================================
  // Vulkan helpers
  // ============================================================================
  /**
   * @brief Find a suitable memory type index for allocating Vulkan memory. The function takes a type filter bitmask, which indicates the allowed memory types based on the resource's memory requirements, and a set of desired memory property flags (e.g., host visible, device local). It queries the physical device's memory properties and iterates through the available memory types to find one that matches both the type filter and the required properties. If a suitable memory type is found, its index is returned. If no suitable memory type is found, an error message is printed and the program exits.
   * @param typeFilter A bitmask where each bit represents a memory type that is suitable based on the resource's memory requirements (e.g., from VkMemoryRequirements.memoryTypeBits).
   * @param properties A bitmask of VkMemoryPropertyFlags that specifies the desired properties of the memory (e.g., VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT).
   * @return The index of a suitable memory type that satisfies both the type filter and the required properties.
   */
  static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(g_ctx.physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
        return i;
    }

    fprintf(stderr, "[Renderer] findMemoryType: no suitable memory type\n");
    exit(EXIT_FAILURE);
  }

  /**
   * @brief Create a Vulkan shader module from SPIR-V bytecode. The function takes a vector of bytes containing the SPIR-V code, fills in a VkShaderModuleCreateInfo structure with the appropriate fields (including the code size and pointer to the code), and calls vkCreateShaderModule to create the shader module. If the creation is successful, the resulting VkShaderModule handle is returned. If there is an error during creation, an error message is printed and the program exits.
   * @param code A vector of bytes containing the SPIR-V shader bytecode.
   * @return The created VkShaderModule handle.
   */
  static VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    vkCreateShaderModule(g_ctx.device, &ci, nullptr, &module);
    
    return module;
  }

  // ============================================================================
  // GLFW callbacks: camera controls
  // ============================================================================
  /**
   * @brief GLFW scroll callback to handle zooming the camera in and out. The function adjusts the camera distance (g_ctx.camDist) based on the vertical scroll offset (yoff), allowing the user to zoom in and out. The camera distance is clamped to a minimum of 1.0f and a maximum of 100.0f to prevent excessive zooming. If ImGui is initialized and wants to capture mouse input (e.g., when hovering over a slider), the function returns early without modifying the camera distance, allowing ImGui to handle the scroll event instead.
   * @param w The GLFW window that received the scroll event (unused).
   * @param xoff The horizontal scroll offset (unused).
   * @param yoff The vertical scroll offset.
   */
  static void scrollCallback(GLFWwindow* /*w*/, double /*xoff*/, double yoff) {
    // Let ImGui consume the event when a widget is hovered (e.g. slider)
    if (g_ctx.imguiInitialized && ImGui::GetIO().WantCaptureMouse) return;
    g_ctx.camDist -= (float)yoff * 0.5f;
    if (g_ctx.camDist < 1.0f)   g_ctx.camDist = 1.0f;
    if (g_ctx.camDist > 100.0f) g_ctx.camDist = 100.0f;
  }

  /**
   * @brief GLFW mouse button callback to handle initiating camera rotation when the left mouse button is pressed. When the left mouse button is pressed, the function sets a flag (g_ctx.mouseDown) to indicate that the mouse is currently dragging, and records the current cursor position (g_ctx.lastMouseX, g_ctx.lastMouseY) to use as a reference for calculating camera rotation in the cursor position callback. If ImGui is initialized and wants to capture mouse input, the function returns early without modifying the mouse state, allowing ImGui to handle the mouse button event instead.
   * @param w The GLFW window that received the mouse button event.
   * @param button The mouse button that was pressed or released.
   * @param action The action performed on the mouse button (GLFW_PRESS, GLFW_RELEASE).
   * @param mods The modifier keys (GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, etc.).
   */
  static void mouseButtonCallback(GLFWwindow* w, int button, int action, int /*mods*/) {
    if (g_ctx.imguiInitialized && ImGui::GetIO().WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      g_ctx.mouseDown = (action == GLFW_PRESS);
      if (g_ctx.mouseDown)
        glfwGetCursorPos(w, &g_ctx.lastMouseX, &g_ctx.lastMouseY);
    }
  }

  /**
   * @brief GLFW cursor position callback to handle camera rotation when the mouse is dragged. If the left mouse button is currently held down (g_ctx.mouseDown), the function calculates the change in cursor position since the last recorded position, and updates the camera angles (g_ctx.camTheta for azimuth and g_ctx.camPhi for elevation) based on the cursor movement. The camera angles are adjusted by a sensitivity factor (0.005f) to control the rotation speed. The elevation angle (camPhi) is clamped to prevent gimbal lock when looking straight up or down. If ImGui is initialized and wants to capture mouse input, the function returns early without modifying the camera angles, allowing ImGui to handle the cursor movement instead.
   * @param w The GLFW window that received the cursor position event (unused).
   * @param xpos The x-coordinate of the cursor position.
   * @param ypos The y-coordinate of the cursor position.
   */
  static void cursorPosCallback(GLFWwindow* /*w*/, double xpos, double ypos) {
    if (!g_ctx.mouseDown) return;
    if (g_ctx.imguiInitialized && ImGui::GetIO().WantCaptureMouse) return;

    float dx = (float)(xpos - g_ctx.lastMouseX) * 0.005f;
    float dy = (float)(ypos - g_ctx.lastMouseY) * 0.005f;
    g_ctx.camTheta -= dx;
    g_ctx.camPhi   += dy;
    // Clamp elevation to avoid gimbal lock
    if (g_ctx.camPhi >  1.5f) g_ctx.camPhi =  1.5f;
    if (g_ctx.camPhi < -1.5f) g_ctx.camPhi = -1.5f;
    g_ctx.lastMouseX = xpos;
    g_ctx.lastMouseY = ypos;
  }

  /**
   * @brief GLFW key callback to handle keyboard input for camera reset and toggling ball tracking. When the Escape key is pressed, the function signals the window to close. When the 'R' key is pressed, the camera parameters are reset to a default side view. When the 'T' key is pressed, a boolean flag (g_ctx.trackBall) is toggled to enable or disable ball tracking, and a message is printed to the console indicating the new state of ball tracking. If ImGui is initialized and wants to capture keyboard input, the function returns early without modifying the camera or tracking state, allowing ImGui to handle the key event instead.
   * @param w The GLFW window that received the key event.
   * @param key The keyboard key that was pressed or released.
   * @param scancode The platform-specific scancode of the key (unused).
   * @param action The action performed on the key (GLFW_PRESS, GLFW_RELEASE).
   * @param mods The modifier keys (GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, etc.) (unused).
   */
  static void keyCallback(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
    if (g_ctx.imguiInitialized && ImGui::GetIO().WantCaptureKeyboard) return;
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_R) {
    
      // Reset camera to default side view
      g_ctx.camTheta = 1.57f;
      g_ctx.camPhi   = 0.30f;
      g_ctx.camDist  = 25.0f;
      g_ctx.centerX  = 10.0f;
      g_ctx.centerY  = 5.5f;
      g_ctx.centerZ  = 0.0f;
    }
    if (key == GLFW_KEY_T) {
    
      // Toggle ball tracking
      g_ctx.trackBall = !g_ctx.trackBall;
      printf("[Renderer] Ball tracking: %s\n", g_ctx.trackBall ? "ON" : "OFF");
    }
  }

  /**
   * @brief Set a 4×4 matrix to a perspective projection matrix. The function takes a pointer to an array of 16 floats (representing a 4×4 matrix in column-major order) and sets it to a perspective projection matrix based on the provided field of view (in radians), aspect ratio, near plane distance, and far plane distance. The resulting matrix will transform 3D coordinates into clip space for perspective rendering, where objects farther from the camera appear smaller. The specific layout of the resulting perspective projection matrix will depend on the Vulkan convention for clip space, where the Z range is [0, 1].
   * @param m A pointer to an array of 16 floats that will be set to the perspective projection matrix. The matrix is in column-major order.
   * @param fovRad The vertical field of view in radians.
   * @param aspect The aspect ratio of the viewport.
   * @param nearP The distance to the near clipping plane.
   * @param farP The distance to the far clipping plane.
    The resulting perspective projection matrix will have the following layout (Vulkan clip space with Z in [0, 1]):
    [ f/aspect  0       0              0            ]
    [ 0         f       0              0            ]
    [ 0         0       far/(near-far) -1           ]
    [ 0         0       (near*far)/(near-far) 0     ]
    where f = 1/tan(fovRad/2)
   */
  static void mat4Perspective(float* m, float fovRad, float aspect, float nearP, float farP) {
    memset(m, 0, 16 * sizeof(float));
    float tanHalf = tanf(fovRad * 0.5f);
    m[0]  = 1.0f / (aspect * tanHalf);
    m[5]  = 1.0f / tanHalf;
    m[10] = farP / (nearP - farP);
    m[11] = -1.0f;
    m[14] = (nearP * farP) / (nearP - farP);
  }

  /**
   * @brief Set a 4×4 matrix to a look-at view matrix. The function takes a pointer to an array of 16 floats (representing a 4×4 matrix in column-major order) and sets it to a view matrix that transforms world coordinates into the camera's view space. The function uses the camera's eye position, the point it is looking at, and the up vector to compute the forward, right, and true up vectors for the camera's coordinate system. The resulting look-at matrix will position and orient the camera in the scene according to these parameters. The specific layout of the resulting look-at matrix will follow the Vulkan convention for view transformations.
   * @param m A pointer to an array of 16 floats that will be set to the look-at view matrix. The matrix is in column-major order.
   * @param eyeX The x-coordinate of the camera's eye position.
   * @param eyeY The y-coordinate of the camera's eye position.
   * @param eyeZ The z-coordinate of the camera's eye position.
   * @param atX The x-coordinate of the point the camera is looking at.
   * @param atY The y-coordinate of the point the camera is looking at.
   * @param atZ The z-coordinate of the point the camera is looking at.
   * @param upX The x-coordinate of the up vector.
   * @param upY The y-coordinate of the up vector.
   * @param upZ The z-coordinate of the up vector.
    The resulting look-at view matrix will have the following layout (Vulkan convention):
    [  rx   ux   -fx   0 ]
    [  ry   uy   -fy   0 ]
    [  rz   uz   -fz   0 ]
    [ -dot(r,eye) -dot(u,eye) dot(f,eye) 1 ]
    where:
    f = normalize(at - eye)       // forward vector
    r = normalize(cross(f, up))   // right vector
    u = cross(r, f)               // true up vector
   */
  static void mat4LookAt(float* m, float eyeX, float eyeY, float eyeZ,
                          float atX, float atY, float atZ,
                          float upX, float upY, float upZ) {
    
    // Forward (camera looks along -Z in right-handed)
    float fx = atX - eyeX, fy = atY - eyeY, fz = atZ - eyeZ;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;
    // Right = forward × up
    float rx = fy * upZ - fz * upY;
    float ry = fz * upX - fx * upZ;
    float rz = fx * upY - fy * upX;
    float rl = sqrtf(rx*rx + ry*ry + rz*rz);
    rx /= rl; ry /= rl; rz /= rl;
    // True up = right × forward
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    // Column-major
    m[0] = rx;  m[4] = ry;  m[8]  = rz;  m[12] = -(rx*eyeX + ry*eyeY + rz*eyeZ);
    m[1] = ux;  m[5] = uy;  m[9]  = uz;  m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
    m[2] = -fx; m[6] = -fy; m[10] = -fz; m[14] =  (fx*eyeX + fy*eyeY + fz*eyeZ);
    m[3] = 0;   m[7] = 0;   m[11] = 0;   m[15] = 1.0f;
  }

  /**
   * @brief Multiply two 4×4 matrices (in column-major order) and store the result in an output matrix. The function takes three pointers to arrays of 16 floats each, representing the output matrix (out), the first input matrix (a), and the second input matrix (b). It performs standard matrix multiplication, where the resulting matrix is computed as out = a * b. The multiplication is done in a way that is compatible with column-major order, which is commonly used in graphics applications and follows the Vulkan convention. The resulting matrix will also be in column-major order.
   * @param out A pointer to an array of 16 floats where the result of the matrix multiplication will be stored. The matrix is in column-major order.
   * @param a A pointer to the first input matrix (array of 16 floats) that will be multiplied. The matrix is in column-major order.
   * @param b A pointer to the second input matrix (array of 16 floats) that will be multiplied. The matrix is in column-major order.
     The multiplication is defined as:
     out[i,j] = sum_k a[i,k] * b[k,j]
     where i and j are the row and column indices of the output matrix, and k iterates over the shared dimension of the input matrices. The function uses temporary storage to compute the result before copying it to the output array.
   */
  static void mat4Multiply(float* out, const float* a, const float* b) {
    float tmp[16];
    for (int c = 0; c < 4; c++) {
      for (int r = 0; r < 4; r++) {
        tmp[c*4 + r] = 0;
        for (int k = 0; k < 4; k++) {
          tmp[c*4 + r] += a[k*4 + r] * b[c*4 + k];
        }
      }
    }
    memcpy(out, tmp, 16 * sizeof(float));
  }

  // ============================================================================
  // Dear ImGui - initialization, shutdown, parameter UI
  // ============================================================================

  /**
   * @brief Initialize Dear ImGui for use with Vulkan and GLFW. The function creates an ImGui context, configures the IO settings to enable keyboard navigation, and applies a dark theme with custom styling. It then initializes the ImGui backends for both GLFW (for windowing and input) and Vulkan (for rendering). The Vulkan initialization requires filling in an ImGui_ImplVulkan_InitInfo structure with details about the Vulkan instance, physical device, logical device, queue family, queue, render pass, swapchain image count, and other parameters. After successful initialization, a flag is set to indicate that ImGui is initialized, and a message is printed to the console.
   */
  static void initImGui(void) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dark theme with custom accent
    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding   = 6.0f;
    style.FrameRounding    = 4.0f;
    style.GrabRounding     = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.Alpha            = 0.92f;

    // GLFW backend
    ImGui_ImplGlfw_InitForVulkan(g_ctx.window, true);

    // Vulkan backend - use DescriptorPoolSize to auto-create a pool
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance        = g_ctx.instance;
    initInfo.PhysicalDevice  = g_ctx.physicalDevice;
    initInfo.Device          = g_ctx.device;
    initInfo.QueueFamily     = g_ctx.graphicsFamily;
    initInfo.Queue           = g_ctx.graphicsQueue;
    initInfo.RenderPass      = g_ctx.renderPass;
    initInfo.MinImageCount   = 2;
    initInfo.ImageCount      = (uint32_t)g_ctx.swapchainImages.size();
    initInfo.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    initInfo.DescriptorPoolSize = 1;  // auto-create descriptor pool
    initInfo.Subpass         = 0;

    ImGui_ImplVulkan_Init(&initInfo);

    g_ctx.imguiInitialized = true;
    printf("[Renderer] Dear ImGui initialized\n");
  }

  /**
   * @brief Shutdown Dear ImGui and clean up resources. The function checks if ImGui was initialized, and if so, it calls the shutdown functions for both the Vulkan and GLFW backends, destroys the ImGui context, and sets the initialization flag to false. This should be called during renderer cleanup to ensure that all ImGui resources are properly released.
   */
  static void shutdownImGui(void) {
    if (!g_ctx.imguiInitialized) return;

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    g_ctx.imguiInitialized = false;
  }

  // ---------------------------------------------------------------------------
  // drawParamsUI - render Dear ImGui parameter panel
  // Returns true when the "Start Simulation" button is pressed.
  // ---------------------------------------------------------------------------
  /**
   * @brief Render the Dear ImGui user interface for displaying and editing simulation parameters. The function creates a panel on the right side of the window that contains controls for adjusting various simulation parameters such as gravity, timestep, damping, collision stiffness, restitution, and friction. The UI can be rendered in either editable mode (where sliders are interactive) or read-only mode (where parameter values are displayed as text). An optional "Start Simulation" button can be shown at the top of the panel, which returns true when pressed. The function also includes tooltips for each parameter to provide additional information to the user when hovering over the controls.
   * @param params A reference to a SimParams structure that contains the simulation parameters to be displayed and edited in the UI.
   * @param editable A boolean flag that indicates whether the parameter sliders should be interactive (true) or read-only (false). In read-only mode, the parameter values are displayed as text instead of sliders.
   * @param showStartButton A boolean flag that indicates whether to show the "Start Simulation" button at the top of the panel. If true, the button will be rendered and the function will return true when it is pressed. If false, the button will not be shown.
   * @return A boolean value that is true if the "Start Simulation" button was pressed during this call, and false otherwise. This allows the caller to detect when the user has initiated the simulation by clicking the button.
    The UI layout includes collapsible sections for "Physics" and "Collision" parameters, with sliders for each parameter. When a slider is hovered, a tooltip is displayed with a description of that parameter. The panel is designed to be user-friendly and informative, allowing users to easily adjust simulation settings before starting the simulation.
   */
  static bool drawParamsUI(SimParams &params, bool editable, bool showStartButton) {
    bool startPressed = false;

    // Panel position: right side of window
    ImGui::SetNextWindowPos(ImVec2((float)g_ctx.winW - 500.0f, 10.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(490.0f, (float)g_ctx.winH - 20.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(430.0f, 200.0f), ImVec2(630.0f, (float)g_ctx.winH));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoResize;
    if (!editable)
      flags |= ImGuiWindowFlags_NoInputs;

    ImGui::Begin("Simulation Parameters", nullptr, flags);

    // ---- Start button (top, prominent) ----
    if (showStartButton) {
      ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.0f, 213.0f / 255.0f, 0.0f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 230.0f / 255.0f, 0.0f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.0f, 180.0f / 255.0f, 0.0f, 1.0f));
      float btnW = ImGui::GetContentRegionAvail().x;
      if (ImGui::Button("START SIMULATION", ImVec2(btnW, 40.0f)))
        startPressed = true;

      ImGui::PopStyleColor(3);
      ImGui::Separator();
      ImGui::Spacing();
    }

    const float MAXSLIDERWIDTH = 200.0f;

    // Helper: slider with tooltip
    #define PARAM_SLIDER_F(label, ptr, lo, hi, txt, fmt) \
      do { \
        if (editable) { \
          float rowX   = ImGui::GetCursorPosX(); \
          float availW = ImGui::GetContentRegionAvail().x; \
          const float minCol = 50.0f; \
          const float gap    = 10.0f; \
          float sliderW = availW - (minCol * 2.0f) - (gap * 3.0f); \
          if (sliderW > MAXSLIDERWIDTH) sliderW = MAXSLIDERWIDTH; \
          \
          ImGui::Text(fmt, (double)(lo)); \
          ImGui::SameLine(rowX + minCol + gap); \
          ImGui::SetNextItemWidth(sliderW); \
          ImGui::SliderFloat(label, ptr, lo, hi, fmt, ImGuiSliderFlags_AlwaysClamp); \
          bool __hovered = ImGui::IsItemHovered(); \
          ImGui::SameLine(rowX + minCol + gap + sliderW + gap); \
          ImGui::Text(fmt, (double)(hi)); \
          ImGui::SameLine(rowX + minCol + gap + sliderW + gap + minCol + gap); \
          ImGui::Text("%s", txt); \
          if (__hovered) ImGui::SetTooltip("%s", label + 2); \
        } else { \
          ImGui::Text("%s: " fmt, label + 2, *(ptr)); \
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", label + 2); \
        } \
      } while(0)

    #define PARAM_SLIDER_I(label, ptr, lo, hi, txt) \
      do { \
        if (editable) { \
          float rowX   = ImGui::GetCursorPosX(); \
          float availW = ImGui::GetContentRegionAvail().x; \
          const float minCol = 50.0f; \
          const float gap    = 10.0f; \
          float sliderW = availW - (minCol * 2.0f) - (gap * 3.0f); \
          if (sliderW > MAXSLIDERWIDTH) sliderW = MAXSLIDERWIDTH; \
          \
          ImGui::Text("%d", (int)(lo)); \
          ImGui::SameLine(rowX + minCol + gap); \
          ImGui::SetNextItemWidth(sliderW); \
          ImGui::SliderInt(label, ptr, lo, hi, "%d", ImGuiSliderFlags_AlwaysClamp); \
          bool __hovered = ImGui::IsItemHovered(); \
          ImGui::SameLine(rowX + minCol + gap + sliderW + gap); \
          ImGui::Text("%d", (int)(hi)); \
          ImGui::SameLine(rowX + minCol + gap + sliderW + gap + minCol + gap); \
          ImGui::Text("%s", txt); \
          if (__hovered) ImGui::SetTooltip("%s", label + 2); \
        } else { \
          ImGui::Text("%s: %d", label + 2, *(ptr)); \
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", label + 2); \
        } \
      } while(0)

    // ---- Snowpack ----
    if (ImGui::CollapsingHeader("Snowpack", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_I("##Snowpack particle count", &params.numParticles, 10000, 1000000, "Particles");
      PARAM_SLIDER_F("##Spawn length along slope (m)", &params.spawnLengthS, 100.0f, 1000.0f, "Spawn Length (m)", "%.1f");
      PARAM_SLIDER_F("##Spawn width across slope (m)", &params.spawnWidthZ, 10.0f, 50.0f, "Spawn Width (m)", "%.1f");
      PARAM_SLIDER_F("##Spawn start position along slope (m)", &params.spawnStartS, 0.0f, 10.0f, "Spawn Start (m)", "%.1f");
      PARAM_SLIDER_F("##Spawn thickness along slope (m)", &params.spawnThicknessN, 0.01f, 0.1f, "Spawn Thickness (m)", "%.2f");
      PARAM_SLIDER_I("##Max captures per frame (0=unlimited)", &params.maxCapturePerFrm, 0, 1000, "Max Capture/frm");
    }

    // ---- Physics ----
    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Gravitational acceleration (m/s^2)", &params.gravity, 1.0f, 30.0f, "Gravity (m/s^2)", "%.1f");
      //PARAM_SLIDER_F("##Timestep (s)", &params.dt, 0.003f, 0.01f, "dt (s)", "%.4f");
      PARAM_SLIDER_F("##Linear velocity damping coefficient", &params.damping, 0.0f, 1.0f, "Damping", "%.3f");
    }

    // ---- Soft-Sphere Collision ----
    if (ImGui::CollapsingHeader("Collision", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Penalty stiffness (k_pen)", &params.stiffness, 0.0f, 1e6f, "Stiffness", "%.0f");
      PARAM_SLIDER_F("##Collision damping (k_damp)", &params.collisionDamping, 0.0f, 100.0f, "Coll. Damping", "%.1f");
      PARAM_SLIDER_F("##Coefficient of restitution (ground)", &params.restitution, 0.0f, 1.0f, "Restitution", "%.2f");
      PARAM_SLIDER_F("##Coulomb friction (ground)", &params.friction, 0.0f, 2.0f, "Friction", "%.2f");
      PARAM_SLIDER_F("##Inter-particle tangential friction", &params.particleFriction, 0.0f, 2.0f, "Part. Friction", "%.2f");
    }

    // ---- Cohesion ----
    if (ImGui::CollapsingHeader("Cohesion", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Cohesion strength (k_coh)", &params.cohesion, 0.0f, 1e5f, "Cohesion", "%.0f");
      PARAM_SLIDER_F("##Cohesion cutoff radius (m)", &params.cohesionRadius, 0.01f, 0.3f, "Coh. Radius (m)", "%.3f");
    }

    // ---- Particle Geometry ----
    if (ImGui::CollapsingHeader("Particle", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Uniform particle radius (m)", &params.particleRadius, 0.01f, 0.005f, "Radius (m)", "%.3f");
      PARAM_SLIDER_F("##Uniform particle mass (kg)", &params.particleMass, 0.1f, 1.0f, "Mass (kg)", "%.4f");
      PARAM_SLIDER_F("##Snow density for radius calc (kg/m^3)", &params.snowDensity, 50.0f, 300.0f, "Density (kg/m^3)", "%.0f");
    }

    // ---- Terrain ----
    if (ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Slope angle (°)", &params.slopeAngleDeg, 30.0f, 60.0f, "Slope Angle (°)", "%.1f");
      PARAM_SLIDER_F("##Along-slope domain length (m)", &params.domainLength, 1.0f, 500.0f, "Domain Length (m)", "%.0f");
    }

    // ---- Ball Drag ----
    if (ImGui::CollapsingHeader("Ball Drag", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Rolling drag coefficient (linear, per s)", &params.rollingDrag, 0.0f, 1.0f, "Rolling Drag", "%.4f");
      PARAM_SLIDER_F("##Aerodynamic drag coefficient (F=coeff*R^2*v^2)", &params.aeroCoeff, 0.0f, 5.0f, "Aero Coeff", "%.3f");
    }
    
    // ---- Sticking Model (Avalanche) ----
    if (ImGui::CollapsingHeader("Sticking Model", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Base logit for sticking probability", &params.stickK0, -10.0f, 10.0f, "K0 (base)", "%.2f");
      PARAM_SLIDER_F("##Wetness contribution to logit", &params.stickK1, -10.0f, 10.0f, "K1 (wetness)", "%.2f");
      PARAM_SLIDER_F("##Velocity penalty per m/s", &params.stickK2, -10.0f, 10.0f, "K2 (velocity)", "%.2f");
      PARAM_SLIDER_F("##Logit boost per metre of ball radius", &params.stickRadiusBoost, -10.0f, 10.0f, "Radius Boost", "%.2f");
      PARAM_SLIDER_F("##Max absolute logit value for sticking sigmoid", &params.logitClamp, 1.0f, 30.0f, "Logit Clamp", "%.1f");
    }

    // ---- Wetness ----
    if (ImGui::CollapsingHeader("Wetness", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Minimum initial wetness", &params.wetnessMin, 0.0f, 1.0f, "Min", "%.2f");
      PARAM_SLIDER_F("##Maximum initial wetness", &params.wetnessMax, 0.0f, 1.0f, "Max", "%.2f");
    }

    // ---- Shell (Dynamic Particles) ----
    if (ImGui::CollapsingHeader("Shell", ImGuiTreeNodeFlags_DefaultOpen)) {
      PARAM_SLIDER_F("##Tether spring stiffness to core (N/m)", &params.shellTetherK, 0.0f, 5000.0f, "Tether K (N/m)", "%.0f");
      PARAM_SLIDER_F("##Tether damping coefficient", &params.shellTetherDamp, 0.0f, 50.0f, "Tether Damp", "%.1f");
      PARAM_SLIDER_F("##Cell size (m)", &params.cellSize, 0.01f, 0.5f, "Cell Size (m)", "%.4f");
    }

    #undef PARAM_SLIDER_F
    #undef PARAM_SLIDER_I

    ImGui::End();

    return startPressed;
  }

  /**
   * @brief Render a dedicated window for camera and scene-position tuning.
   *
   * The window is intentionally separate from simulation parameters so users can
   * quickly iterate on view presets and fine controls while the scene is running.
   */
  static void drawCameraSceneUI(void) {
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 320.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(300.0f, 220.0f), ImVec2(520.0f, 520.0f));

    if (!ImGui::Begin("Camera & Scene")) {
      ImGui::End();
      return;
    }

    ImGui::Checkbox("Track Ball", &g_ctx.trackBall);
    if (g_ctx.trackBall)
      ImGui::TextDisabled("Center follows snowball while simulation runs");
    ImGui::Checkbox("Show Ball", &g_ctx.showBall);

    ImGui::Separator();
    ImGui::Text("Camera");
    ImGui::SliderFloat("camTheta", &g_ctx.camTheta, -6.2832f, 6.2832f, "%.4f");
    ImGui::SliderFloat("camPhi", &g_ctx.camPhi, -1.5f, 1.5f, "%.4f");
    ImGui::SliderFloat("camDist", &g_ctx.camDist, 1.0f, 180.0f, "%.1f");
    ImGui::SliderFloat("FOV (deg)", &g_ctx.fovDeg, 20.0f, 100.0f, "%.1f");
    ImGui::SliderFloat("Near Plane", &g_ctx.nearPlane, 0.01f, 5.0f, "%.2f");
    ImGui::SliderFloat("Far Plane", &g_ctx.farPlane, 50.0f, 600.0f, "%.0f");

    if (g_ctx.nearPlane >= g_ctx.farPlane - 0.01f)
      g_ctx.nearPlane = g_ctx.farPlane - 0.01f;

    ImGui::Separator();
    ImGui::Text("Scene Center (look-at)");
    ImGui::SliderFloat("Center X", &g_ctx.centerX, -100.0f, 200.0f, "%.2f");
    ImGui::SliderFloat("Center Y", &g_ctx.centerY, -50.0f, 100.0f, "%.2f");
    ImGui::SliderFloat("Center Z", &g_ctx.centerZ, -100.0f, 100.0f, "%.2f");

    ImGui::End();

    //drawBillboardUI(); // TODO comment to hide
  }

  // ============================================================================
  // Public API: runPreSimUI - pre-simulation parameter editing loop
  // ============================================================================
  /**
   * @brief Run the pre-simulation user interface loop for editing parameters. This function enters a loop that continues until the user either starts the simulation by pressing the "Start Simulation" button or closes the window. Inside the loop, it polls for GLFW events, renders the ImGui interface for editing simulation parameters, and performs a simplified Vulkan rendering pass that includes the background, slope, ground, and ImGui interface (but no particles). The camera is static during this pre-simulation phase. The function returns true when the "Start Simulation" button is pressed, allowing the caller to proceed with running the simulation using the edited parameters.
   * @param params A reference to a SimParams structure that will be edited by the user through the ImGui interface. The function will display the current values of the parameters and allow the user to modify them before starting the simulation.
   * @return A boolean value that is true if the "Start Simulation" button was pressed during the UI loop, and false if the window was closed without starting the simulation. This allows the caller to determine whether to proceed with running the simulation or to exit based on user input.
     The function handles rendering a simplified scene during the pre-simulation phase, which includes drawing the background sky, textured slope, and ground plane. It also builds a static camera view and projection matrix for this phase. The Vulkan command buffer is recorded with these elements and the ImGui interface, but does not include any particle rendering since the simulation has not started yet.
   */
  bool runPreSimUI(SimParams &params) {
    if (!g_ctx.window || !g_ctx.imguiInitialized) return false;

    printf("[Renderer] Pre-simulation UI - edit parameters and press START\n");

    bool started = false;

    while (!glfwWindowShouldClose(g_ctx.window)) {
      glfwPollEvents();

      // ImGui frame
      ImGui_ImplVulkan_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Draw editable parameter panel with Start button
      started = drawParamsUI(params, true, true);
      drawCameraSceneUI(); // TODO comment to hide

      ImGui::Render();

      // Build MVP (static camera for pre-sim view)
      float eyeX = g_ctx.centerX + g_ctx.camDist * cosf(g_ctx.camPhi) * sinf(g_ctx.camTheta);
      float eyeY = g_ctx.centerY + g_ctx.camDist * sinf(g_ctx.camPhi);
      float eyeZ = g_ctx.centerZ + g_ctx.camDist * cosf(g_ctx.camPhi) * cosf(g_ctx.camTheta);
      float view[16], proj[16];
      mat4LookAt(view, eyeX, eyeY, eyeZ, g_ctx.centerX, g_ctx.centerY, g_ctx.centerZ, 0.0f, 1.0f, 0.0f);
      float aspect = (float)g_ctx.winW / (float)g_ctx.winH;
      mat4Perspective(proj, g_ctx.fovDeg * 3.14159265f / 180.0f, aspect, g_ctx.nearPlane, g_ctx.farPlane);
      proj[5] *= -1.0f;
      mat4Multiply(g_ctx.mvp, proj, view);

      // Vulkan draw
      uint32_t fi = g_ctx.currentFrame;
      vkWaitForFences(g_ctx.device, 1, &g_ctx.inFlight[fi], VK_TRUE, UINT64_MAX);
      vkResetFences(g_ctx.device, 1, &g_ctx.inFlight[fi]);

      uint32_t imageIndex;
      VkResult res = vkAcquireNextImageKHR(g_ctx.device, g_ctx.swapchain, UINT64_MAX,
                                           g_ctx.imageAvailable[fi], VK_NULL_HANDLE, &imageIndex);
      if (res == VK_ERROR_OUT_OF_DATE_KHR) continue;

      vkResetCommandBuffer(g_ctx.commandBuffers[fi], 0);

      // Record simplified command buffer (background + slope + ground + ImGui, no particles)
      {
        VkCommandBuffer cmd = g_ctx.commandBuffers[fi];
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkClearValue clearValues[2]{};
        clearValues[0].color        = {{0.05f, 0.05f, 0.12f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass        = g_ctx.renderPass;
        rpBegin.framebuffer       = g_ctx.framebuf[imageIndex];
        rpBegin.renderArea.offset = {0, 0};
        rpBegin.renderArea.extent = g_ctx.swapchainExtent;
        rpBegin.clearValueCount   = 2;
        rpBegin.pClearValues      = clearValues;

        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        // Background sky
        if (g_ctx.bgPipeline) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.bgPipeline);
          vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        // Push constants
        float pushData[17];
        memcpy(pushData, g_ctx.mvp, 16 * sizeof(float));
        pushData[16] = 1.0f;

        // Textured slope
        if (g_ctx.slopePipeline && g_ctx.slopeSurfBuf && g_ctx.slopeSurfVertexCount > 0) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.slopePipeline);
          vkCmdPushConstants(cmd, g_ctx.slopePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.slopePipelineLayout,
                                  0, 1, &g_ctx.slopeDescSet, 0, nullptr);
          VkDeviceSize off = 0;
          vkCmdBindVertexBuffers(cmd, 0, 1, &g_ctx.slopeSurfBuf, &off);
          vkCmdDraw(cmd, (uint32_t)g_ctx.slopeSurfVertexCount, 1, 0, 0);
        }

        // Ground plane  
        if (g_ctx.triPipeline && g_ctx.slopeBuffer && g_ctx.slopeVertexCount > 0) {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx.triPipeline);
          vkCmdPushConstants(cmd, g_ctx.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
          VkDeviceSize off = 0;
          vkCmdBindVertexBuffers(cmd, 0, 1, &g_ctx.slopeBuffer, &off);
          vkCmdDraw(cmd, (uint32_t)g_ctx.slopeVertexCount, 1, 0, 0);
        }

        // ImGui overlay
        ImDrawData* drawData = ImGui::GetDrawData();
        if (drawData) {
          ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
      }

      // Submit + present
      VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      submitInfo.waitSemaphoreCount   = 1;
      submitInfo.pWaitSemaphores      = &g_ctx.imageAvailable[fi];
      submitInfo.pWaitDstStageMask    = &waitStage;
      submitInfo.commandBufferCount   = 1;
      submitInfo.pCommandBuffers      = &g_ctx.commandBuffers[fi];
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores    = &g_ctx.renderFinished[fi];

      vkQueueSubmit(g_ctx.graphicsQueue, 1, &submitInfo, g_ctx.inFlight[fi]);

      VkPresentInfoKHR presentInfo{};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores    = &g_ctx.renderFinished[fi];
      presentInfo.swapchainCount     = 1;
      presentInfo.pSwapchains        = &g_ctx.swapchain;
      presentInfo.pImageIndices      = &imageIndex;

      vkQueuePresentKHR(g_ctx.presentQueue, &presentInfo);

      g_ctx.currentFrame = (fi + 1) % MAX_FRAMES_IN_FLIGHT;

      if (started) break;
    }

    if (started) {
      // Drain Vulkan pipeline - ensures all submitted frames from the
      // pre-sim UI are fully complete before the simulation loop takes over.
      vkDeviceWaitIdle(g_ctx.device);

      // Recompute derived slope values from user-edited slopeAngleDeg
      float rad = params.slopeAngleDeg * (float)M_PI / 180.0f;
      params.slopeSin = sinf(rad);
      params.slopeCos = cosf(rad);
      g_ctx.slopeSin  = params.slopeSin;
      g_ctx.slopeCos  = params.slopeCos;

      // Store the final user-edited params so renderFrame can display
      // them as a disabled (read-only) overlay during simulation.
      g_ctx.simParams = params;
      g_ctx.simRunning = true;

      printf("[Renderer] User pressed START - simulation begins\n");
      return true;
    }

    printf("[Renderer] Window closed during pre-sim UI\n");
    return false;
  }
  #include "vulkan_renderer_billboard.inl"
#endif // ENABLE_VULKAN