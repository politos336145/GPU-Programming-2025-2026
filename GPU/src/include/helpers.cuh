#ifndef HELPERS_CUH
  #define HELPERS_CUH
  
  #define CUDA_CHECK(call)                                                                          \
    do {                                                                                            \
      cudaError_t err = (call);                                                                     \
      if (err != cudaSuccess) {                                                                     \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                         \
      }                                                                                             \
    }                                                                                               \
    while(0)

  /**
   * @brief Compute grid size (number of blocks) for given total threads and block size.
   *        It's inline because it is included in multiple .cu files, and we want to avoid multiple definitions.
   * 
   * @param N           Total number of threads needed (e.g. number of particles).
   * @param blockSize   Number of threads per block (e.g. 256).
   * 
   * @return Number of blocks needed to cover N threads with given blockSize.
   */
  inline int gridSize(int N, int blockSize) { return (N + blockSize - 1) / blockSize; }

  // ============================================================================
  // __constant__ SimParams d_params
  //
  // All .cu files that include this header MUST be compiled in one translation
  // unit so that every kernel sees the same d_params instance.  This is
  // ensured by gpu_unity.cu, which #includes grid.cu, simulation.cu, and
  // snowpack.cu.  main.cu is compiled separately but has no device code that
  // reads d_params.
  // ============================================================================
  __constant__ SimParams d_params;

  // ============================================================================
  // Spatial hash for compact grid (prime-number mixing, Teschner et al. 2003)
  // Produces a well-distributed hash from 3-D cell coordinates, avoiding the
  // systematic Z-axis collisions of the naive linear-index & mask approach.
  // ============================================================================
  /**
   * @brief Compute hash for 3D cell coordinates (cx, cy, cz) with given mask.
   *        Uses prime-number multiplication and XOR to mix the coordinates,
   *        then applies the mask to fit into the hash table size.
   * 
   * @param cx    Cell x coordinate (integer).
   * @param cy    Cell y coordinate (integer).
   * @param cz    Cell z coordinate (integer).
   * @param mask  Hash table size minus one (must be a power of two).
   */
  __device__ __forceinline__ uint32_t cellHash3D(int cx, int cy, int cz, int mask) {
    uint32_t h = (uint32_t)cx * 73856093u
               ^ (uint32_t)cy * 19349663u
               ^ (uint32_t)cz * 83492791u;
    h ^= h >> 16;          // extra bit-mix so low bits are not purely XOR
    return h & (uint32_t)mask;
  }
    
  // ============================================================================
  // Quaternion rotation helpers (unit quaternion assumed)
  // ============================================================================

  /**
   * @brief Rotate vector v by quaternion q (v' = q * v * q^{-1}).
   * 
   * @param qw  Quaternion scalar component.
   * @param qx  Quaternion x imaginary component.
   * @param qy  Quaternion y imaginary component.
   * @param qz  Quaternion z imaginary component.
   * @param vx  Input vector x component.
   * @param vy  Input vector y component.
   * @param vz  Input vector z component.
   * @param ox  Output rotated vector x component.
   * @param oy  Output rotated vector y component.
   * @param oz  Output rotated vector z component.
   */
  __device__ __forceinline__ void quatRotateVec(
      float qw, float qx, float qy, float qz,
      float vx, float vy, float vz,
      float &ox, float &oy, float &oz)
  {
    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
  }

  /**
   * @brief Inverse-rotate vector v to body frame (v_local = q^* * v * q).
   * 
   * @param qw  Quaternion scalar component.
   * @param qx  Quaternion x imaginary component.
   * @param qy  Quaternion y imaginary component.
   * @param qz  Quaternion z imaginary component.
   * @param vx  Input vector x component (world frame).
   * @param vy  Input vector y component (world frame).
   * @param vz  Input vector z component (world frame).
   * @param ox  Output vector x component (body frame).
   * @param oy  Output vector y component (body frame).
   * @param oz  Output vector z component (body frame).
   */
  __device__ __forceinline__ void quatInvRotateVec(
      float qw, float qx, float qy, float qz,
      float vx, float vy, float vz,
      float &ox, float &oy, float &oz)
  {
    // conjugate: (w, -x, -y, -z)
    float cx = -qx, cy = -qy, cz = -qz;
    float tx = 2.0f * (cy * vz - cz * vy);
    float ty = 2.0f * (cz * vx - cx * vz);
    float tz = 2.0f * (cx * vy - cy * vx);
    ox = vx + qw * tx + (cy * tz - cz * ty);
    oy = vy + qw * ty + (cz * tx - cx * tz);
    oz = vz + qw * tz + (cx * ty - cy * tx);
  }
#endif // HELPERS_CUH