## Table of Contents

1. [Architecture](#1-architecture)
2. [Physics Model](#2-physics-model)
3. [GPU Pipeline](#3-gpu-pipeline)
4. [CPU Optimizations (SSE / SIMD + OpenMP)](#4-cpu-optimizations-sse--simd--openmp)
5. [GPU Optimizations](#5-gpu-optimizations)
6. [Project Structure](#6-project-structure)
7. [Requirements](#7-requirements)
8. [Configuration](#8-configuration)
9. [Rendering](#9-rendering)
10. [References](#10-references)

---

## 1. Architecture

The simulation is split into three cooperating subsystems:

```
┌──────────────────────────────────────────────────────────────┐
│  SNOWPACK (static)        SHELL (dynamic)        CORE (RB)   │
│  ──────────────────       ──────────────         ──────────  │
│  SoA on device            ParticleSystem SoA     SnowballSt  │
│  posXYZ, radius,          posXYZ, velXYZ,        posXYZ,     │
│  mass, wetness,           forceXYZ, mass,        velXYZ,     │
│  alive[AVAILABLE/         radius, state,         quat,       │
│        CONSUMED]          wetness,               omega,      │
│  count (fixed)            attachLocal*           mass, R     │
│                           capacity (dynamic)                 │
│                           count (grows)                      │
└──────────────────────────────────────────────────────────────┘
         │  capture kernel          │ tether spring
         └──────────────────────────┘
```

| Subsystem | Data | Mutability | Update |
|-----------|------|------------|--------|
| **Snowpack** | `Snowpack` SoA - positions, radius, mass, wetness, alive flag (`AVAILABLE`/`CONSUMED`) | Static positions; `alive` flag toggled on capture; sorted once at init by `posX` for range-limited capture | `captureFromSnowpackKernel` marks consumed particles |
| **Shell** | `ParticleSystem` SoA - full dynamics (pos, vel, force, mass, radius, state, wetness, attachLocal) | Grows as snowpack particles are captured (capacity doubles dynamically, starting at 256); full DEM physics each frame | Gravity, tether, collision, integration, ground |
| **Core** | `SnowballState` struct - position, velocity, quaternion, angular velocity, mass, radius, `capturedParticleCount` | Updated on host each frame | Rigid-body rolling on slope with momentum conservation, rolling drag, and aerodynamic drag |

### 1.1. Data Flow per Frame

```
[Capture] snowpack→shell (range-limited via binary search on sorted posX)
    → [Core update] mass/radius/vel (deferred readback, 1-frame latency)
    → [ForcesTether] gravity+damping+tether (fused)
    → [Grid] spatial hash (parallel stream)  → [Collision] neighbor DEM + cohesion
    → [IntegrateGround] semi-implicit Euler + slope constraint (fused)
    → [Core RB] rigid-body slope dynamics (host)
```

For small shell counts ($N \leq 2048$), stages 3–6 are replaced by a single **mega-fused kernel** (`shellPhysicsFusedKernel`) that performs forces + brute-force $O(N^2)$ collision + integration in one launch, keeping all intermediate data in registers.

---

## 2. Physics Model

The simulation employs the **Discrete Element Method (DEM)** — a numerical method in which granular material (sand, snow, gravel…) is represented as a set of discrete particles, each modelled as a small sphere with mass and radius, that interact via pairwise contact forces. When two particles overlap (penetration), repulsive forces (spring + damping) and friction are applied to push them apart, reproducing granular behaviour without solving continuous equations. The variant used here is a **soft-sphere contact model** (penalty-based) with Coulomb friction, **without** persistent contact history between pairs: tangential springs are not tracked across frames, and the friction impulse is recomputed independently at each timestep from the current relative velocity.

### 2.1. Terrain Geometry

The simulation takes place on an infinite inclined plane:

- **Slope normal**: $\hat{N} = (\sin\theta,\ \cos\theta,\ 0)$
- **Downhill tangent**: $\hat{T} = (\cos\theta,\ -\sin\theta,\ 0)$
- **Plane equation**: $\sin\theta \cdot x + \cos\theta \cdot y = H$
- Default angle: 30°, default height offset $H = 30$ m

The slope origin at $x=0$ is at $y = H / \cos\theta$. All particle positions are in world coordinates; the slope-local curvilinear coordinate $s$ maps to world as:

$$x = s \cos\theta + n \sin\theta, \quad y = \frac{H}{\cos\theta} - s \sin\theta + n \cos\theta$$

where $n$ is the offset along the surface normal.

### 2.2. Snowpack Initialization

Particles are placed on a rectangular band on the slope surface parameterised by curvilinear coordinates $(s, z, n)$:
- $s \in [\text{startS},\ \text{startS} + \text{lengthS}]$ — along slope (downhill)
- $z \in [-W/2,\ +W/2]$ — lateral
- $n \in [0,\ \text{thickness}]$ — normal to surface (stacking layers)

A regular grid is constructed within each layer: the number of columns is $\text{cols} = \lfloor\sqrt{N_\text{layer} \cdot W / L}\rfloor$, and rows are derived accordingly. Each particle receives $\pm 30\%$ jitter within its grid cell to break regularity. Per-particle wetness $w \in [\text{wMin}, \text{wMax}]$ is assigned randomly.

After initialization, the snowpack is **sorted once by ascending `posX`** using CUB radix sort (GPU) or `std::sort` (CPU). This enables per-frame **binary search** on the host to find only the narrow $X$-range around the ball, reducing the capture kernel's launch domain by 50–100×.

### 2.3. Snowball Core — Rigid Body

The snowball rolls under gravity along the slope. The downhill acceleration is:

$$a_{\text{slope}} = g \sin\theta$$

applied in the tangent direction $\hat{T} = (\cos\theta, -\sin\theta, 0)$.

Two drag forces oppose the motion:

1. **Rolling drag** (linear, models rolling friction and surface deformation):
   $$\mathbf{v} \leftarrow \mathbf{v} \cdot (1 - k_{\text{roll}} \cdot \Delta t)$$

2. **Aerodynamic drag** (quadratic, proportional to frontal area $R^2$):
   $$\Delta v_{\text{aero}} = \frac{C_{\text{aero}} \cdot R^2 \cdot |\mathbf{v}|}{m} \cdot \Delta t$$
   applied opposite to velocity, clamped so drag never reverses motion in one step. At large $R$ this becomes the dominant braking term, giving physically bounded terminal velocity even as the ball grows.

The ball is constrained to the slope surface: signed distance $d = \sin\theta \cdot x + \cos\theta \cdot y - H$; if $d < R$, the ball is pushed out along $\hat{N}$ and the normal velocity component is zeroed.

**Rolling-without-slip angular velocity**:

$$\boldsymbol{\omega} = \frac{1}{R}(\hat{N}_{\text{slope}} \times \mathbf{v})$$

expanded as: $\omega_x = \frac{\cos\theta \cdot v_z}{R}$, $\omega_y = \frac{-\sin\theta \cdot v_z}{R}$, $\omega_z = \frac{\sin\theta \cdot v_y - \cos\theta \cdot v_x}{R}$.

**Quaternion integration** preserves orientation:

$$\mathbf{q}' = \mathbf{q} + \tfrac{1}{2}\Delta t\;(0, \boldsymbol{\omega}) \otimes \mathbf{q}$$

with renormalization each frame to prevent drift. The ball starts very small ($R_0 = 0.03$ m, $m_0 = \frac{4}{3}\pi R_0^3 \rho_{\text{snow}}$) and grows dramatically via capture.

### 2.4. Probabilistic Capture (Snowpack → Shell)

The capture is implemented in `captureFromSnowpackKernel`. Each frame, the host performs a binary search on the pinned sorted array `h_snowpackPosX` to identify the relevant window $[\text{spLo}, \text{spHi})$; the kernel processes only particles in this range (parameters `spOffset`, `spCount`), reducing work from $O(N_{\text{snowpack}})$ to a geometrically relevant subset. The capture counters reside in a packed 8-byte device buffer `[int count | float mass]`, reset each frame with `cudaMemsetAsync`.

**Per-thread algorithm:**

1. **Early exit**: if `alive[idx] != AVAILABLE` → skip (already consumed).

2. **Distance test**: compute $d = |\mathbf{x}_{\text{sp}} - \mathbf{x}_{\text{ball}}|$ and compare with the capture radius:
   $$r_{\text{capture}} = R + r_p + 0.3\text{ m (margin)}$$
   If $d > r_{\text{capture}}$ → skip.

3. **Relative velocity**: since the snowpack is static, $\mathbf{v}_{\text{rel}} = -\mathbf{v}_{\text{ball}}$. Compute $|\mathbf{v}_{\text{rel}}|$.

4. **Logit computation** (probability score):
   $$\text{logit} = k_0 + k_1 \cdot w - k_2 \cdot |\mathbf{v}_{\text{rel}}| + k_R \cdot R$$
   where:
   - $k_0$ (default 1.0): base bias
   - $k_1 \cdot w$ (default $k_1 = 7.11$): wetter snow → more adhesion
   - $-k_2 \cdot |\mathbf{v}_{\text{rel}}|$ (default $k_2 = 1.1$): faster ball → less adhesion
   - $k_R \cdot R$ (default $k_R = 7.5$): **avalanche feedback** — larger ball → easier capture

   The logit is **clamped** to $[-L_{\max}, +L_{\max}]$ (default $\pm 15.0$) for numerical stability.

5. **Probabilistic test**:
   $$P_{\text{stick}} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
   A pseudo-random number $u \in [0, 1)$ is generated via a **deterministic hash** (murmurhash variant, seeded with particle index and frame number — not `curand`). If $u \geq P_{\text{stick}}$ → skip.

6. **Lock-free slot reservation** (atomic, no mutex):
   ```cuda
   int slot = atomicAdd(d_captureCount, 1);
   if (maxCapture > 0 && slot >= maxCapture) return;  // per-frame cap
   if (shellN + slot >= shellCapacity) return;         // shell full
   ```
   Guarantees at most `maxCapturePerFrm` (default 150) writes. Late threads exit without side effects. The classic lock-free pattern: `atomicAdd` returns the old value (the reserved slot) before other threads increment it.

7. **Writing into shell** at position `shell[shellN + slot]`:
   - **Position**: projected onto ball surface: $\mathbf{x}_{\text{ball}} + \hat{n} \cdot (R + r_p)$
   - **Velocity**: synchronized with surface point: $\mathbf{v}_{\text{ball}} + \boldsymbol{\omega} \times \mathbf{r}$
   - **Forces**: zeroed
   - **Mass, radius, wetness**: copied from snowpack particle
   - **`attachLocal`**: direction vector rotated into body frame via `quatInvRotateVec()`
   - Mass accumulated via `atomicAdd(d_captureMass, mass)` — only **after** passing cap and capacity checks

8. **Marking**: `alive[idx] = CONSUMED`.

**Three stability mechanisms** prevent runaway growth:
- **Logit clamp**: bounds the sigmoid input to $[-15, +15]$, preventing saturation at extreme values
- **Per-frame cap**: at most `maxCapturePerFrm` particles per frame, limiting growth rate
- **Lock-free atomic slot reservation**: threads beyond the cap or shell capacity exit cleanly without side effects

**Deferred readback.** The capture result is read back via asynchronous D2H copy (pinned memory + `cudaMemcpyAsync`). The host processes the result at the top of the **next** frame (1-frame latency, physically negligible). When `actualNew < d_captureCount` (some threads exceeded the cap or capacity), the captured mass is scaled proportionally:
$$m_{\text{credited}} = m_{\text{total}} \cdot \frac{\text{actualNew}}{\text{d\_captureCount}}$$
This avoids inflating `ball.radius` and triggering a runaway feedback loop.

### 2.5. Shell Particle Forces

Each shell particle accumulates forces in the `forceTetherKernel` (fused kernel):

1. **Gravity**: $\mathbf{F}_g = m \cdot (0, -g, 0)$
2. **Mass-proportional linear damping**: $\mathbf{F}_d = -k_d \cdot m \cdot \mathbf{v}$
3. **Tether to core**: spring-damper pulling each particle toward its anchor point on the ball surface:
   $$\mathbf{F}_{\text{tether}} = K(\mathbf{x}_{\text{target}} - \mathbf{x}) + D(\mathbf{v}_{\text{target}} - \mathbf{v})$$
   where the target position is $\mathbf{x}_{\text{target}} = \mathbf{x}_{\text{ball}} + \text{rotate}(\mathbf{q}, \hat{d}_{\text{local}}) \cdot (R + r_p)$, and the target velocity accounts for the ball's angular velocity: $\mathbf{v}_{\text{target}} = \mathbf{v}_{\text{ball}} + \boldsymbol{\omega} \times \mathbf{r}$.

The collision forces are computed in a separate kernel (`neighborCollisionKernel`) and **accumulated** on top of the tether forces (single-writer per thread, no `atomicAdd`):

4. **Soft-sphere collision** (DEM penalty): for overlapping particles ($d < r_i + r_j$):
   $$\mathbf{F}_{\text{contact}} = \max(k_{\text{pen}} \cdot \delta - k_{\text{damp}} \cdot v_n,\; 0) \cdot \hat{n}$$
   with Coulomb tangential friction: $|\mathbf{F}_t| \leq \mu_p |\mathbf{F}_n|$
5. **Cohesion**: short-range attraction for $r_i + r_j < d < r_{\text{cut}}$:
   $$\mathbf{F}_{\text{coh}} = -k_{\text{coh}}(1 - d/r_{\text{cut}}) \hat{n}$$

### 2.6. Integration

Semi-implicit Euler (symplectic):

$$\mathbf{v}^{n+1} = \mathbf{v}^n + \frac{\mathbf{F}}{m}\Delta t, \quad \mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1}\Delta t$$

### 2.7. Ground Collision

The inclined-plane constraint applies to each shell particle:
- **Signed distance**: $d = \sin\theta \cdot x + \cos\theta \cdot y - H$
- **Penetration resolution**: if $d < r_p$, push particle out along slope normal by $(r_p - d)$
- **Normal velocity**: $v_n = v_x \sin\theta + v_y \cos\theta$
- **Normal restitution** (only if $v_n < 0$): $\Delta v_n = -(1 + e) \cdot v_n$
- **Coulomb friction**: clamp tangential velocity magnitude by $\mu \cdot |\Delta v_n|$; if friction exceeds tangential speed, tangential velocity is zeroed

### 2.8. Momentum Conservation on Capture

When particles are captured, the ball's velocity is scaled to conserve linear momentum:

$$\mathbf{v}_{\text{ball}}' = \frac{m_{\text{old}}}{m_{\text{old}} + \Delta m} \cdot \mathbf{v}_{\text{ball}}$$

Ball radius is derived from total mass and snow density:

$$R = \sqrt[3]{\frac{3 m}{4 \pi \rho_{\text{snow}}}}$$

---

## 3. GPU Pipeline

### 3.1. Large-N Path ($N_{\text{shell}} > 2048$)

Seven-stage pipeline using two CUDA streams with fork/join synchronization:

| # | Stage | Kernel / Function | Stream | Block Size | Timing Label |
|---|-------|-------------------|--------|------------|--------------|
| 1 | **Capture** | `captureFromSnowpackKernel` (range-limited) | `simStream` | 256 | `captureMs` |
| 2 | **Core mass update** | Host code (deferred readback + momentum conservation) | host | — | — |
| 3 | **Forces + Tether** (fused) | `forceTetherKernel` | `simStream` | 256 | `shellForcesMs` |
| 4 | **Grid build** | `clearPrevCellsKernel` + `computeHashKernel` + CUB radix sort + `buildCellRangesKernel` | `gridStream` | 256 | `shellGridMs` |
| 5 | **Collision** | `neighborCollisionKernel` | `simStream` | 512 | `shellCollisionMs` |
| 6 | **Integrate + Ground** (fused) | `integrateGroundKernel` | `simStream` | 256 | `shellIntegrateMs` |
| 7 | **Core RB** | `updateSnowball()` | host | — | `coreUpdateMs` |

**Multi-stream architecture**:

```
simStream :  Capture → ForcesTether ─── evFork ───────────── wait(evGridDone) → Collision → IntGround
                                          │                         ▲
gridStream:                               └── wait(evFork) → Grid ─ evGridDone
traceStream:  (optional depending on the parameter --trace-interval) GPU trace capture
```

Stages 3 and 4 execute in parallel on separate streams. Fork/join uses `cudaEventDisableTiming` (lightweight events, no timing overhead). Stage 5 (collision) waits for both the grid build and the force computation to complete before launching.

### 3.2. Small-N Path ($N_{\text{shell}} \leq 2048$)

For small shell counts, the overhead of building the spatial grid (5–8 GPU commands: clear + hash + CUB sort + ranges + collision) exceeds the benefit. A single **mega-fused kernel** replaces stages 3–6:

| # | Stage | Kernel / Function | Stream | Block Size | Timing Label |
|---|-------|-------------------|--------|------------|--------------|
| 1 | **Capture** | `captureFromSnowpackKernel` | `simStream` | 256 | `captureMs` |
| 2 | **Core mass update** | Host code (deferred readback) | host | — | — |
| 3–6 | **Shell Physics (fused)** | `shellPhysicsFusedKernel` | `simStream` | 256 | `shellFusedMs` |
| 7 | **Core RB** | `updateSnowball()` | host | — | `coreUpdateMs` |

`shellPhysicsFusedKernel` fuses gravity + damping + tether + brute-force $O(N^2)$ collision + integration + ground collision into a single kernel launch. Forces stay entirely in registers (no global memory round-trip). Collision uses shared-memory tiling with `__syncthreads()` (block-wide N-body pattern, 256-element tiles).

### 3.3. Block Size and `__launch_bounds__`

| Kernel | Block Size | `__launch_bounds__` | Rationale |
|--------|-----------|---------------------|-----------|
| `neighborCollisionKernel` | **512** | `(512, 3)` | Warp-cooperative tile sized for $\text{MAX\_WARPS} = 16 = 512/32$; 3 resident blocks/SM → 75% occupancy |
| `forceTetherKernel` | 256 | `(256, 8)` | 26 regs → 100% occupancy; 16+ blocks → full SM coverage |
| `integrateGroundKernel` | 256 | `(256, 8)` | 31 regs → 100% occupancy |
| `captureFromSnowpackKernel` | 256 | `(256, 5)` | Range-limited launches (small effective N); ~62.5% occupancy (register-limited) |
| `shellPhysicsFusedKernel` | 256 | `(256, 5)` | Small-N fused path; shared-memory tiling with 256-element tiles |
| `bruteForceCollisionKernel` | 256 | `(256, 5)` | Small-N standalone collision (shared-memory tiling) |
| `computeHashKernel` | 256 | none | Lightweight; 100% occupancy |
| `buildCellRangesKernel` | 256 | none | Lightweight; 100% occupancy |
| `initSnowpackKernel` | 1024 (256 on Jetson) | none | $N = 1{,}000{,}000$ → 977 blocks even with 1024; reduced to 256 on SM 5.3 (Jetson Nano) due to register/shared memory constraints |

Per-kernel timing is collected via `cudaEvent` pairs. Timing events are **gated**: only recorded every `TIMING_INTERVAL = 64` frames (or when logging), eliminating per-frame `cudaEventSynchronize` stalls in headless mode.

**Kernel fusion rationale**:
- **`forceTetherKernel`** merges the former `applyForcesKernel` (gravity + damping) and `shellCoreTetherKernel` (spring-damper to core) into a single kernel. This eliminates one kernel launch and one global-memory round-trip for force arrays. The fused kernel reads each particle's position and velocity once and writes a single combined force vector.
- **`integrateGroundKernel`** merges the former `integrateKernel` (semi-implicit Euler) and `groundCollisionKernel` (slope constraint). After computing the new position and velocity, ground collision is resolved in-register before writing back — saving one kernel launch and one pos/vel memory round-trip.

---

## 4. CPU Optimizations (SSE / SIMD + OpenMP)

The project includes two CPU builds that mirror the GPU pipeline:

- **SISD** (`AngrySanta_SISD`): scalar baseline with all compiler optimizations disabled (`-O0`, `/Od`), no inlining, no auto-vectorization, no loop unrolling. Serves as a performance reference for measuring GPU and SIMD speedups.
- **SIMD** (`AngrySanta_SIMD`): SSE/NEON vectorized with OpenMP multi-threading. Full compiler optimizations enabled.

Both CPU builds share the same simulation loop (`runSimulation()` in `simulation_CPU.cpp`) and the same physics pipeline (capture → forces+tether → grid → collision → integrate+ground → core update). The SIMD build overrides the per-kernel functions with vectorized implementations.

### 4.1. Architecture Detection

The SIMD build supports both x86 (SSE2) and ARM (NEON) via compile-time detection:

```cpp
#if defined(__x86_64__) || defined(_M_X64) || ...
  #include <immintrin.h>     // SSE / SSE2 intrinsics
  #define CPU_SIMD_X86 1
#elif defined(__aarch64__) || ...
  #include <arm_neon.h>      // NEON intrinsics
  #define CPU_SIMD_NEON 1
#endif
```

### 4.2. Vectorized Kernels

Each physics function processes 4 particles per iteration (128-bit SIMD lanes):

| Function | SIMD Pattern | Scalar Remainder |
|----------|-------------|-----------------|
| `initSnowpack` | SSE: `_mm_loadu_ps` / `_mm_storeu_ps`; NEON: `vld1q_f32` / `vst1q_f32` | `initSnowpack_scalar()` for $N \% 4$ tail |
| `forcesTether` | SSE: `_mm_load_ps` / `_mm_mul_ps`; quaternion rotation fully vectorized | `forcesTether_scalar()` for tail |
| `integrateGround` | SSE: fused velocity update + ground collision check | `integrateGround_scalar()` for tail |
| `buildGrid` (hash) | SSE: `_mm_cvttps_epi32` for cell coordinate computation | `buildGrid_hashScalar()` for tail |

Each vectorized function uses aligned loads (`_mm_load_ps`) for SoA arrays allocated with 16-byte alignment (via `ALLOC` macro in `memory_CPU.h`), and unaligned loads for gather-scatter patterns. The scalar remainder loop handles the $N \% 4$ tail elements.

### 4.3. OpenMP Parallelization

When compiled with OpenMP (`-DHAS_OPENMP`), the SIMD loops use `#pragma omp parallel for schedule(static)` for data-parallel distribution across CPU cores. This applies to initialization, force computation, and integration loops.

### 4.4. Memory Allocation

The SIMD build uses 16-byte aligned allocation for packed loads/stores:

```cpp
#ifdef SIMD
  #ifdef _MSC_VER
    #define ALLOC(bytes)  _aligned_malloc((bytes), 16)
    #define FREE(ptr)     _aligned_free(ptr)
  #else
    #define ALLOC(bytes)  aligned_alloc(16, (bytes))
    #define FREE(ptr)     free(ptr)
  #endif
#else
  #define ALLOC(bytes)  malloc(bytes)
  #define FREE(ptr)     free(ptr)
#endif
```

---

## 5. GPU Optimizations

This section provides a comprehensive analysis of all GPU-level optimizations
implemented in the project, discussing principles, alternatives, trade-offs, and
quantitative impact. The optimizations are divided into two phases: foundational
GPU techniques and performance engineering refinements.

**Target hardware**: GTX 960M (SM 5.0, 5 SMs, 128 CUDA cores/SM, 2 GB GDDR5, 1 MB L2)
and Jetson Nano (SM 5.3, 1 SM, 128 CUDA cores, 256 KB L2).

### 5.1. Memory Coalescing (SoA Layout)

A GPU warp (32 threads) issues a single memory transaction. If threads 0–31 access
contiguous addresses (`addr, addr+4, addr+8, …`), the hardware merges them into a
single 128-byte transaction — this is **coalescing**. Scattered accesses (stride > 1
element) can waste up to 97% of bus bandwidth.

| Layout | Bytes/warp per `posX` load | Transactions | Efficiency |
|--------|---------------------------|--------------|-----------|
| **SoA** (separate `float *posX, *posY, …`) | 128 B (32 contiguous floats) | 1 | 100% |
| **AoS** (`struct Particle { float x,y,z,vx,vy,vz,… }`) | 32 × 48 = 1536 B scattered | up to 12 | ~10% |

The spatial grid introduces **indirect** accesses (`posX[particleIndex[s]]`), which
are not perfectly coalesced. However, after the CUB radix sort, particles within the
same cell have nearby indices — loads are *nearly* coalesced and benefit from L2 cache
reuse. The inner loop of `neighborCollisionKernel` accesses neighbors through
indirection (`int j = __ldg(&particleIndex[s])`), which is inherent to data-dependent
neighbor search; the L2 cache (1 MB) mitigates the cost by reusing particle data
already loaded by previous threads in the same cell.

### 5.2. Shared Memory Tiling

Shared memory is on-chip SRAM (~5 cycles latency, ~1.3 TB/s) shared among threads
within the same block, compared to global memory (~300–500 cycles, ~30 GB/s on
Maxwell). **Tiling** loads a block of elements from global memory into `__shared__`,
synchronizes, then has all threads read from shared memory with a reuse ratio of $N/B$.

**Warp-cooperative tile.** The `neighborCollisionKernel` uses shared memory in a
**warp-cooperative** fashion: each warp loads a tile of `WARP_SIZE = 32` neighbors
into shared memory, then each thread reads from the shared tile. The per-warp tile
is declared with dimensions `[MAX_WARPS][WARP_SIZE + 1]`, where
`MAX_WARPS = 16 = 512 / 32` and the `+1` padding avoids shared memory bank conflicts:

```cuda
__shared__ float s_px[MAX_WARPS][WARP_SIZE + 1];
__shared__ float s_py[MAX_WARPS][WARP_SIZE + 1];
__shared__ float s_pz[MAX_WARPS][WARP_SIZE + 1];
__shared__ float s_r[MAX_WARPS][WARP_SIZE + 1];
__shared__ int   s_idx[MAX_WARPS][WARP_SIZE + 1];
__shared__ int   s_state[MAX_WARPS][WARP_SIZE + 1];
```

**Why `__syncwarp()` instead of `__syncthreads()`.** The previous kernel used
`__syncthreads()` combined with symmetric iteration and `atomicAdd`. This caused a
deadlock: when the block has more threads than active particles, excess threads execute
`return` before the barrier, violating the CUDA specification that requires *all*
threads in a block to reach the same `__syncthreads()`. `__syncwarp()` is
**warp-scoped** — it synchronizes only the 32 threads of the current warp, and all 32
always participate in the tile load (the `lane < tileLen` guard is a predicated branch,
not an early `return`).

**Why block-wide tiling is not used.** With the spatial grid, neighbor list lengths
vary per cell and access is indirect. A block-wide tile would require padding and
complex synchronization. The 32-element warp-tile fits naturally: each cell-batch has
at most a few dozen neighbors, handled in 1–2 tiles without padding.

**L2 cache coverage.** The working set of 4,000 steady-state shell particles
(8 fields × 4 B = 32 B/particle → 128 KB total) is only 12.5% of the L2 cache (1 MB
on GTX 960M), so data not in the shared tile is served efficiently by L2. If the
project scales beyond $N > 100{,}000$ on SM ≥ 7.0 (Volta+), the working set (3.2 MB)
would exceed L2 and a block-wide tile would reduce DRAM traffic.

### 5.3. Atomic Reduction Strategy

On **Maxwell (SM 5.0/5.3)**, there is no dedicated hardware for `atomicAdd` on FP32
— the compiler generates a **compare-and-swap** (CAS) loop that under contention can
execute 5–10 retry iterations per operation.

**Previous approach (symmetric iteration + atomicAdd).** Each pair was computed once
(`j > i`), requiring 6 `atomicAdd` per pair (forceX/Y/Z for both particles). The CAS
stalls exceeded the compute savings from halving the iterations.

**Current approach (full non-symmetric iteration, zero atomicAdd).** Every thread $i$
iterates over *all* neighbors $j \neq i$, accumulating forces entirely in registers.
The only write is a single direct store to `forceX/Y/Z[i]` — thread $i$ is the sole
writer, so no atomic is needed:

```cuda
float fi_x = 0.0f, fi_y = 0.0f, fi_z = 0.0f;
for (every neighbor j != i) {
    fi_x += fx;  fi_y += fy;  fi_z += fz;  // register accumulation
}
forceX[i] += fi_x;  // direct store, zero contention
```

| Aspect | Full iteration (current) | Symmetric (`j > i`) |
|---|---|---|
| Pairs computed per $N$ neighbors | $N$ (each pair twice) | $N/2$ (each pair once) |
| `atomicAdd` per particle | **0** | 3 (i-side) + 3/contacting j |
| Cost per force write | ~4 cycles (store) | ~20–60 cycles (CAS retry) |
| Warp divergence | No | Yes (`j > i` skip) |

The only remaining atomics are in `captureFromSnowpackKernel` for global counters
(1 int + 1 float per capture event, ~0–50 per frame) — negligible contention.

### 5.4. Occupancy Tuning

**Occupancy** = active warps / max warps per SM. On SM 5.0/5.3: max = 64 warps
(2048 threads). Occupancy is limited by registers (65,536/SM), shared memory
(48 KB/SM), and block size (max 32 blocks/SM).

**Per-kernel occupancy table** (from `ptxas -v` output):

| Kernel | Regs/thread | Block size | `__launch_bounds__` | Blocks/SM | Threads/SM | Occupancy | Limiting factor |
|--------|------------|-----------|---------------------|----------|-----------|-----------|----------------|
| `neighborCollisionKernel` | 40 | 512 | `(512, 3)` | 3 | 1536 | **75%** | Shared memory (~12.4 KB/block × 3 ≈ 37 KB < 48 KB) |
| `forceTetherKernel` | 27 | 256 | `(256, 8)` | 8 | 2048 | **100%** | None |
| `integrateGroundKernel` | 24 | 256 | `(256, 8)` | 8 | 2048 | **100%** | None |
| `captureFromSnowpackKernel` | 48 | 256 | `(256, 5)` | 5 | 1280 | **62.5%** | Registers |
| `shellPhysicsFusedKernel` | 37 | 256 | `(256, 5)` | 5 | 1280 | **62.5%** | Registers |
| `bruteForceCollisionKernel` | 38 | 256 | `(256, 5)` | 5 | 1280 | **62.5%** | Registers |
| `computeHashKernel` | ~17 | 256 | — | 8 | 2048 | **100%** | None |
| `buildCellRangesKernel` | ~7 | 256 | — | 8 | 2048 | **100%** | None |

All kernels report **zero register spill** (confirmed via `ptxas` output), meaning
all local variables stay in physical SM registers rather than being evicted to local
memory (DRAM-backed, ~300 cycle latency).

**SM coverage vs occupancy.** The API `cudaOccupancyMaxPotentialBlockSize` returns the
block size maximizing occupancy per SM (typically 1024), but ignores **SM coverage** —
how many SMs receive at least one block. With $N = 4{,}000$ and block = 1024: only
$\lceil 4000/1024 \rceil = 4$ blocks for 5 SMs → 1 SM idle (20% hardware wasted).
With block = 256: $\lceil 4000/256 \rceil = 16$ blocks → all 5 SMs active. The block
size of **512** for `neighborCollisionKernel` is dictated by the tile layout
(`MAX_WARPS = 16 = 512/32`); fewer threads would waste shared memory. Other kernels
use **256** for full SM coverage.

**Portability (Jetson Nano, 1 SM).** Coverage is not an issue with 1 SM; the
constraint is maximizing occupancy within that single SM.

**Why not 100% occupancy everywhere.** Four kernels operate below 100% occupancy.
In each case, forcing 100% would require cutting the per-thread register count
enough to fit more resident blocks, which would cause **register spill** to local
memory (DRAM-backed, ~300 cycle latency per spill load/store) — a net performance
loss. The specific bottlenecks are:

| Kernel | Occupancy | Why 100% is not achievable without regression |
|--------|-----------|-----------------------------------------------|
| `neighborCollisionKernel` | 75% | Dual constraint: 40 regs × 512 = 20,480 regs/block → $\lfloor 65536/20480 \rfloor = 3$ blocks; shared memory: 3 × 12.4 KB = 37 KB < 48 KB, but 4 × 12.4 KB = 49.6 KB > 48 KB. Reaching 4 blocks would require both reducing registers to 32 (causing spill) and shrinking the shared tile (losing cooperative prefetch). |
| `captureFromSnowpackKernel` | 62.5% | 48 regs/thread — the highest in the project. The kernel computes sigmoid (`expf`), hash PRNG, 3D distance, quaternion rotation (`quatInvRotateVec`), and conditional atomic writes. Reaching 100% would require 32 regs/thread (a 33% cut), causing massive spill in a kernel that is already range-limited to ~500–2,000 active threads per frame — occupancy is irrelevant at such low launch counts. |
| `shellPhysicsFusedKernel` | 62.5% | 37 regs + shared memory for brute-force tiling (8 arrays × 257 × 4 B ≈ 8 KB/block). At 5 blocks: 40 KB shared (within budget). At 6+: exceeds 48 KB. Used only for $N \leq 2048$ where compute dominates — more warp-level parallelism would not help. |
| `bruteForceCollisionKernel` | 62.5% | 38 regs + shared tiling, same constraint as the fused kernel above. |

**Design principle: zero spill > 100% occupancy.** Occupancy measures how many
warps are ready to hide memory latency, but this matters only for **memory-bound**
kernels. For compute-bound kernels or kernels whose working set fits in cache,
keeping variables in registers (zero spill) outweighs the benefit of additional
concurrent warps. A single spill store costs ~300 cycles (DRAM round-trip), while
one fewer warp costs only a few scheduling cycles. All kernels in the project
report **0 spill stores/loads** in `ptxas` output — this is by design.

### 5.5. Single Precision (FP32)

Consumer GPUs have an **FP64:FP32 ratio of 1:32** on SM 5.0/5.3 — a `double`
instruction is 32× slower than `float`. With timestep $\Delta t = 5\text{ ms}$, the
Euler integration truncation error ($O(\Delta t^2) \approx 2.5 \times 10^{-5}$)
dominates FP32 rounding ($\epsilon \approx 6 \times 10^{-8}$) by three orders of
magnitude.

**Rules applied in the project:**

1. **All literals are `f`-suffixed.** In CUDA C++, `1.0` without suffix is `double`;
   `nvcc` does not optimize the `float→double→float` promotion chain.
2. **Math functions are `f`-suffixed**: `sqrtf`, `fabsf`, `floorf`, `expf`, `fminf`,
   `fmaxf`.
3. **`M_PI` is defined as `float`**: `3.14159265358979323846f` in `types.h`.
4. **`rsqrtf` pattern**: computes $1/\sqrt{x}$ in a single SFU cycle, then
   `dist = dist2 * invDist` saves one SFU operation vs `sqrtf` + division.

**`--use_fast_math`.** This flag enables `__fdividef`, `__expf`, flush-to-zero for
denormals, and approximate square root. The error is 1–2 ULP (unit in the last place),
acceptable for this simulation.

### 5.6. `__restrict__` and `__ldg()`

**`__restrict__` (no-alias hint).** Tells the compiler that the pointer does not alias
any other pointer, enabling load/store reordering and register caching of loaded values.

**`__ldg()` (read-only data cache).** On Kepler+ (SM ≥ 3.5), `__ldg()` routes loads
through the **texture/read-only cache**, separate from the L1 used for read-write data.
This provides two independent cache paths:

```
                 ┌──────────────────┐
                 │   Global Memory  │
                 └────────┬─────────┘
                          │
                 ┌────────┴─────────┐
                 │     L2 Cache     │ (1 MB, shared across all SMs)
                 └────────┬─────────┘
                      ┌───┴───┐
             ┌────────┤  SM   ├────────┐
             │        └───────┘        │
     ┌───────┴───────┐       ┌─────────┴────────┐
     │  L1 / Shared  │       │  Read-Only Cache  │
     │  R/W data     │       │  __ldg() loads    │
     └───────────────┘       └──────────────────┘
```

All shell-physics and grid kernels use explicit `__ldg()` for read-only inputs
(positions, radii, state, grid indices) and standard loads for read-write data
(forces). `captureFromSnowpackKernel` uses `const __restrict__` pointers instead of
explicit `__ldg()` calls; on SM ≥ 3.5 with `-O3`, the compiler routes these loads
through the read-only cache automatically. This doubles effective cache bandwidth
without conflicts.

**Why velocities are NOT in the shared tile.** Adding `s_vx/vy/vz` to the warp tile
would increase shared memory from ~10.6 KB to ~17.0 KB per block. At 3 blocks, this
exceeds the 48 KB limit (51 KB), reducing occupancy from 75% to 50%. Velocities are
instead loaded **on-demand via `__ldg()`** only when contact is detected (`dist < sumR`),
which is rare (~2–5 contacts out of 10–50 neighbors per cell):

```cuda
if (dist < sumR) {
    float vrel_x = __ldg(&velX[j]) - vi_x;  // on-demand, not tiled
}
```

### 5.7. Constant Memory

`SimParams` (~152 bytes) resides in `__constant__` memory, defined in `helpers.cuh`
and shared across all `.cu` files via Unity Build (`gpu_unity.cu`):

```cuda
__constant__ SimParams d_params;
```

Constant memory occupies a dedicated **64 KB** cache that **broadcasts** when all
threads in a warp read the same address (~5 cycles for the entire warp) and does not
consume registers. Physics parameters (`gravity`, `dt`, `stiffness`, `damping`, etc.)
are read identically by all threads — a perfect use case. Updated once at startup via
`cudaMemcpyToSymbol`.

### 5.8. Kernel Fusion

Two pairs of kernels were fused to eliminate global memory round-trips:

| Before | After | Savings |
|--------|-------|---------|
| `applyForcesKernel` + `shellCoreTetherKernel` | **`forceTetherKernel`** | Eliminates 1 round-trip forceX/Y/Z |
| `integrateKernel` + `groundCollisionKernel` | **`integrateGroundKernel`** | Eliminates 1 round-trip pos/vel |

Fusion keeps intermediate results in **registers** rather than writing to global memory
and re-reading.

### 5.9. Spatial Hashing + CUB Radix Sort

Collision detection goes from $O(N^2)$ to $O(N \cdot k)$ via uniform-grid spatial hashing:

1. **`computeHashKernel`:** each particle computes `(cx,cy,cz)` → hash using **prime multiplication** (Teschner et al. 2003): `h = cx*73856093 ⊕ cy*19349663 ⊕ cz*83492791` + additional bit-mixing, then **`& hashTableMask`** to map into the compact hash table.
2. **CUB `DeviceRadixSort::SortPairs`**: sorts `(hash, particleIndex)` with
   pre-allocated temporary buffer; limited to the necessary number of sort bits
   (`sortBits = ceil(log2(hashTableSize) / 4) × 4`) to minimize passes
3. **`clearPrevCellsKernel`:** clears *only* cells used in the previous frame (scatter-clear using `cellHashAlt`, avoids full-array `memset` — from ~75 MB to a few KB per frame).
4. **`buildCellRangesKernel`**: finds `cellStart[h]` / `cellEnd[h]` for each hash cell

**Compact hash table.** Instead of allocating `cellStart/cellEnd` for all domain cells
(~9.8 M at full resolution), modular hashing with table size
`nextPowerOf2(capacity × 8)` is used — ~32 K entries (~256 KB vs ~75 MB).

**CUB vs Thrust.** CUB radix sort has less overhead and does not depend on Thrust
internals.

### 5.10. Sorted Snowpack + Binary Search Range-Limited Capture

The snowpack is static (positions never change). Sorting it once at initialization by
`posX` enables a **binary search** on the CPU at each frame to find the interval
$[\text{lo}, \text{hi})$ of particles within capture distance from the snowball.
The capture kernel launches only on this subset.

**Sort (once, at init in `snowpack.cu`):** CUB `DeviceRadixSort::SortPairs` on
`(posX, index)` pairs, followed by a gather kernel to reorder all other SoA fields.
A single temporary buffer of $N \times 4$ bytes minimizes peak memory.

**Binary search (per frame, in `main.cu`):**

```cpp
float captureR = ball.radius + params.particleRadius + 0.3f;
int spLo = std::lower_bound(h_snowpackPosX, h_snowpackPosX + N, ball.posX - captureR)
         - h_snowpackPosX;
int spHi = std::upper_bound(h_snowpackPosX, h_snowpackPosX + N, ball.posX + captureR)
         - h_snowpackPosX;
launchCaptureFromSnowpack(..., spLo, spHi - spLo);
```

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Threads launched | 100,000 | ~500–2,000 | **50–100×** |
| Capture time (ms) | 0.291 | ~0.11 | **2.6×** |

The host-side binary search cost is negligible (~1 µs on a sorted array). The pinned
copy `h_snowpackPosX` is one-shot at initialization.

### 5.11. Mega-Fused Kernel for Small-N Shell Physics

With small shells (≤ 2048 particles, typical at $N = 100$K snowpack), the computational
cost of physics kernels is minimal but the **GPU command launch overhead** is fixed
(~10–15 µs per command on the GTX 960M). The standard pipeline requires ~10 GPU
commands per frame (forceTether + grid build + collision + integrateGround + events),
adding ~100–150 µs of pure overhead — over 50% of frame time when actual compute is
below 200 µs.

**`shellPhysicsFusedKernel`** replaces the entire pipeline with a single kernel:

```
Gravity + Damping → Tether → Brute-Force O(N²) → Euler → Ground
         ↑                                                   ↑
    in registers ← forces NEVER transit through global memory → in registers
```

| Aspect | Separate pipeline | Mega-fused kernel |
|--------|-------------------|-------------------|
| GPU commands per frame | ~10 | **1** |
| Global memory round-trips for forces | 2 (write + read) | **0** |
| Launch overhead (GTX 960M) | ~100–150 µs | **~15 µs** |
| Grid build | Yes (5–8 commands) | **No** |

**Adaptive threshold:** the fused path is used when `shellN ≤ 2048`
(`BF_COLLISION_THRESHOLD`). At $N = 2048$, brute-force collision is $O(N^2) = 4$M
iterations, still fast (~0.3 ms on GTX 960M). Above 2048, the grid-based
$O(N \cdot k)$ path with $k \approx 20$ becomes more efficient.

### 5.12. Pinned Memory for Async Copies

**Problem found via Nsight Systems**: 30–40 ms spikes on the GPU caused by
`cudaMemcpyAsync` to **pageable** (stack) memory. The CUDA specification states that
async copy to non-pinned memory degrades to synchronous — the driver inserts a
device-wide synchronization.

**Fix**: all host buffers used for async readback are allocated with `cudaMallocHost`
(pinned memory): capture counters, sorted `posX` for binary search, and trace buffers.
Result: truly asynchronous DMA copy; synchronization only at stream level.

**Consolidated capture counters.** The count (int) and mass (float) are packed into a
single contiguous 8-byte device buffer, reducing per-frame GPU commands from 4
(2 memsets + 2 memcpys) to 2 (1 memset + 1 memcpy) — saving ~20–30 µs per frame.

### 5.13. Multi-Stream Pipeline

Physics kernels run on dedicated **non-blocking CUDA streams** with fork/join via
lightweight events (`cudaEventDisableTiming`):

```
simStream :  ForcesTether ─── evFork ──────────────── wait(evGridDone) → Collision → IntGround
                                │                           ▲
gridStream:                     └── wait(evFork) → Grid ── evGridDone
traceStream:                    (async D2H copies for trace snapshots)
```

`simStream` runs ForcesTether, then waits for the grid to complete before launching
Collision and IntegrateGround. `gridStream` runs the grid build (hash → sort → ranges)
in parallel with ForcesTether.

### 5.14. Gated Event Recording

`cudaEventRecord()` is a GPU command with ~10–15 µs driver overhead on the GTX 960M.
With up to 14 event records per frame (start + end for 7 measurements), the overhead
is substantial.

**Fix**: events are recorded only every `TIMING_INTERVAL = 64` frames (or when logging
or rendering is active). The 63 non-profiled frames save ~12 event records each,
reducing average overhead by ~11 GPU commands/frame (~110–165 µs).

Stream synchronization events (`evFork`, `evGridDone`, `evCaptureReady`) are always
recorded since they are required for correctness.

### 5.15. Divergence Reduction and ALU Optimization

**`rsqrtf` pattern.** All collision kernels compute normalized distance vectors using
`rsqrtf` directly: `invDist = rsqrtf(dist2)` (1 SFU op, ~8 cycles) followed by
`dist = dist2 * invDist` (1 FMA, ~5 cycles), saving 1 SFU operation compared to
`sqrtf(dist2)` + `1.0f / dist` (2 SFU ops).

**Tangential friction with `rsqrtf`.** The Coulomb friction computation uses a
squared-magnitude threshold (`vt2 > 1e-16f`) and `rsqrtf(vt2)` to avoid computing
`sqrtf` + division — saving 1 SFU operation and 1 branch.

**`fmaxf` for branchless clamping.** Contact force clamping
(`f_mag = fmaxf(f_spring - f_damp, 0.0f)`) maps to a single hardware instruction
(`FMNMX`, 1 cycle) on Maxwell, eliminating warp divergence from conditional branches.

### 5.16. Hash-Based PRNG

The capture kernel uses a hash-based PRNG instead of `curand`:

```cuda
uint32_t h = (uint32_t)idx ^ (uint32_t)(frame * 2654435761u);
h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
float rnd = (float)(h & 0x00FFFFFFu) / (float)0x01000000u;
```

**Zero state overhead for capture**: `curand` requires ~48 bytes/thread of state in
global memory. The murmurhash variant uses only the thread index and frame number as
input — no per-thread state. The per-frame seed variation avoids repeating patterns
across frames. Note that `curand` is still used in `initSnowpackKernel` (called once
at startup to randomize snowpack placement), but the per-frame capture kernel avoids
it entirely.

### 5.17. Quaternion Rotation Without Matrix

The tether and capture kernels apply rotations using the Rodrigues quaternion formula:

```cuda
// v' = v + 2w(q×v) + 2(q×(q×v))
float tx = 2.0f * (qy*vz - qz*vy);
ox = vx + qw*tx + (qy*tz - qz*ty);
```

15 FMA operations vs 27 for classic 3×3 matrix multiplication. Also avoids constructing
a temporary 3×3 matrix (9 floats in registers).

### 5.18. Unity Build

All GPU `.cu` files (`grid.cu`, `simulation.cu`, `snowpack.cu`) are `#include`d into
a single translation unit `gpu_unity.cu`. This shares the `__constant__ SimParams
d_params` declaration across all kernels without separate compilation and avoids
device-link overhead.

### 5.19. `__launch_bounds__` on All Kernels

The `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` attribute
instructs `ptxas` to allocate registers such that at least `minBlocksPerMultiprocessor`
blocks can be resident per SM. Initial conservative values (50% occupancy) were
incrementally tuned to the final values shown in the occupancy table (Section 5.4).
All kernels report **0 spill stores/loads** in the `ptxas` output.

### 5.20. Memory Hierarchy Summary

| Level | Usage | Size |
|-------|-------|------|
| **Registers** | All local variables, force accumulators, positions/velocities per thread | 24–48 per thread |
| **`__constant__`** | `SimParams d_params` — broadcast to all threads | ~152 B |
| **Shared memory** | Warp-cooperative tile in `neighborCollisionKernel` (6 arrays × [16][33] × 4 B) | 12.4 KB/block |
| **Read-only cache** | `__ldg()` loads for immutable data (positions, radii, state, grid indices) | Hardware-managed |
| **L2 cache** | Working set of positions/velocities for grid-based collision | 1 MB (SM 5.0) / 256 KB (SM 5.3) |
| **Global memory** | SoA arrays (16 float arrays × N) | GDDR5 |

---

## 6. Project Structure

```
Angry Santa/
├── Project Assignement 2025-26.pdf
├── .gitignore
├── .vscode/
├── README.md                                         
│
├── docs/
│     ├── TECHNICAL_README.md                         # This file
│     ├── 01 - Pitch/                                 # Delivery 01: Project pitch
│     ├── 02 - Proposal/                              # Delivery 02: Proposal document
│     ├── 03 - Report/                                # Delivery 03: Final report
│     └── 04 - Presentation/                          # Delivery 04: Presentation slides
│
├── GPU/                                              # NVIDIA CUDA implementation (accelerated version)
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── gpu_unity.cu                          # Unity build compilation unit
│     │     ├── main.cu                               # Entry point, simulation loop, CLI parsing
│     │     ├── simulation.cu                         # GPU host functions + wrappers
│     │     ├── grid.cu                               # Spatial grid (hash, CUB sort, cell ranges)
│     │     ├── snowpack.cu                           # Snowpack init + probabilistic capture kernel
│     │     ├── include/
│     │     │     ├── simulation.cuh                  # GPU simulation declarations
│     │     │     ├── grid.cuh                        # Grid host function declarations
│     │     │     ├── snowpack.cuh                    # Snowpack host function declarations
│     │     │     ├── kernel.cuh                      # All __global__ kernel declarations
│     │     │     ├── helpers.cuh                     # CUDA_CHECK, constant memory, helpers
│     │     └── renderers/
│     │           ├── vulkan_renderer.cu              # Vulkan + CUDA interop renderer
│     │           ├── vulkan_renderer.h               # Renderer API
│     │           ├── vulkan_renderer_billboard.inl   # Billboard rendering implementation
│     │           ├── imgui_iml_all.cpp               # ImGui implementation (GLFW + Vulkan)
│     │           ├── stb_image_impl.cpp              # stb_image implementation for texture loading
│     │           ├── stb_image.h                     # stb_image header
|     |           ├── imgui/                          # Dear ImGui library for UI (submodule or copy)
│     │           ├── shaders/                        # GLSL + SPIR-V shaders
│     │           └── textures/                       # Texture images
│     ├── win/
│     │     ├── build.ps1                             
│     │     └── profile_nsys.ps1                      # Nsight Systems profiling
│     └── linux/
│           ├── build.sh                              
│           ├── profile_ncu.sh                        # Nsight Compute profiling
│           └── profile_nsys.sh                       # Nsight Systems profiling                                
│
├── SIMD/                                             # OpenMP multi-threaded SIMD version (SSE-vectorized)
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── main.cpp
│     │     └── simulation.cpp
│     ├── win/
│     │     └── build.ps1
│     └── linux/
│           └── build.sh
│
├── SISD/                                             # Single-threaded CPU reference implementation
│     ├── CMakeLists.txt
│     ├── src/
│     │     ├── main.cpp
│     │     └── simulation.cpp
│     ├── win/
│     │     └── build.ps1
│     └── linux/
│           └── build.sh
│
├── benchmark/
│     ├── benchmark.ps1                               # Windows scaling benchmark script
│     └── benchmark.sh                                # Linux scaling benchmark script
│
└── shared/                                           # Shared code across all implementations
      ├── cli.cpp                                     # CLI argument parsing
      ├── global.cpp                                  # Global functions used by all versions
      ├── logging.cpp                                 # CSV + stdout logging
      ├── simulation_CPU.cpp                          # Common CPU simulation routines
      └── include/
            ├── cli.h                                 # CLI parsing declarations
            ├── global.h                              # Global function declarations
            ├── logging.h                             # Logging function declarations
            ├── memory_CPU.h                          # CPU memory management declarations
            ├── profiling.h                           # Profiling utilities declarations
            ├── simulation_CPU.h                      # CPU simulation declarations
            └── types.h                               # Common types and constants
```

### 6.1. Key Files

| File | Purpose |
|------|---------|
| `shared/include/types.h` | All struct definitions: `SimParams` (~40 fields), `SnowballState`, `ParticleSystem`, `Snowpack`, `GridData`, `KernelTimings`, enums `ShellParticleState` and `SnowpackParticleState` |
| `shared/include/global.h` | Common host declarations: `defaultParams()`, `initSnowball()`, `updateSnowball()`, `computeGridDimensions()`, `updateCoreMassAfterCapture()`, `setParamsFromCLI()`, `printSimulationSummary()` |
| `shared/global.cpp` | Implementations of shared host functions (used by all three builds to eliminate code duplication) |
| `GPU/src/simulation.cu` | GPU shell kernels: `forceTetherKernel`, `neighborCollisionKernel`, `bruteForceCollisionKernel`, `shellPhysicsFusedKernel`, `integrateGroundKernel`; memory management (`allocateParticleSystem`, `growShell`) |
| `GPU/src/include/simulation.cuh` | GPU launch wrapper declarations with `cudaStream_t` parameters |
| `GPU/src/include/helpers.cuh` | `CUDA_CHECK` macro, `gridSize()`, `__constant__ SimParams d_params`, `cellHash3D()`, quaternion rotation device functions |
| `GPU/src/snowpack.cu` | `initSnowpackKernel` (3D band placement with curand), `captureFromSnowpackKernel` (probabilistic capture with sigmoid + atomic append), `sortSnowpackByPosX` (CUB radix sort + gather) |
| `GPU/src/grid.cu` | `computeHashKernel` + CUB radix sort + `clearPrevCellsKernel` + `buildCellRangesKernel`; compact hash table allocation |
| `GPU/src/gpu_unity.cu` | Manual unity build: `#include "grid.cu"`, `"simulation.cu"`, `"snowpack.cu"` to share `d_params` |
| `GPU/src/main.cu` | GPU entry point: CLI parsing, dual-stream pipeline, gated timing, deferred capture readback, dynamic shell growth, logging, trace snapshots, Vulkan integration |
| `shared/simulation_CPU.cpp` | CPU simulation loop (`runSimulation`), scalar kernel implementations (shared by SISD and SIMD as remainder loops), grid build, neighbor collision |
| `shared/include/profiling.h` | Header-only `ProfilingStats` — accumulates min/avg/max per kernel, prints summary with bottleneck identification |

---

## 7. Requirements

### 7.1. Hardware
- **GPU build**: NVIDIA GPU with compute capability **5.0 or higher** (Maxwell through Ada Lovelace). Minimum 2 GB GPU memory (default config uses ~50 MB).
- **CPU builds**: any x86-64 or ARM64 processor. SIMD build benefits from SSE2 (x86) or NEON (ARM).

### 7.2. Software
- **CUDA Toolkit** 10.2+ (11.x+ recommended for built-in CUB)
- **CMake** 3.10+ (3.18+ recommended for `CMAKE_CUDA_ARCHITECTURES`)
- **C++14** compiler (g++ 7+, MSVC 2017+)
- **Ninja** or **Make** build backend
- **OpenMP** (optional, for SIMD multi-threading)
- **Vulkan SDK** + **GLFW 3.x** (optional, for real-time rendering)

---

## 8. Configuration

### 8.1. SimParams Defaults (from `defaultParams()` in `global.cpp`)

#### 8.1.1. Physics
```cpp
gravity           = 9.81f;     // m/s²
dt                = 0.0033f;   // timestep (s)
damping           = 0.05f;     // mass-proportional velocity damping (k_d)
stiffness         = 30000.0f;  // soft-sphere penalty stiffness (k_pen)
collisionDamping  = 50.0f;     // collision damping (k_damp)
cohesion          = 72450.0f;  // cohesion strength (k_coh)
cohesionRadius    = 0.06f;     // cohesion cutoff (m)
particleRadius    = 0.02f;     // uniform radius (m)
particleMass      = 0.1f;      // per-particle mass (kg)
restitution       = 0.15f;     // ground bounce factor (e)
friction          = 0.14f;     // ground Coulomb friction (μ_ground)
particleFriction  = 0.04f;     // inter-particle friction (μ_p)
snowDensity       = 200.0f;    // kg/m³
```

#### 8.1.2. Slope
```cpp
slopeAngleDeg     = 30.0f;
slopeHeight       = 30.0f;     // plane offset (m)
domainLength      = 150.0f;    // max along-slope travel
```

#### 8.1.3. Capture / Sticking
```cpp
stickK0           = 1.0f;      // base logit → ~73% at v=0, w=1
stickK1           = 7.11f;     // wetness boost in logit
stickK2           = 1.1f;      // velocity penalty (per m/s of |v_rel|)
stickRadiusBoost  = 7.5f;      // avalanche feedback (logit boost per metre of ball radius)
logitClamp        = 15.0f;     // max absolute logit value for sigmoid stability
maxCapturePerFrm  = 150;       // max particles captured per frame (growth rate limiter)
wetnessMin        = 0.0f;      // min initial wetness [0,1]
wetnessMax        = 1.0f;      // max initial wetness [0,1]
```

#### 8.1.4. Shell
```cpp
shellTetherK      = 2717.0f;   // tether spring stiffness (N/m)
shellTetherDamp   = 50.0f;     // tether damping coefficient
```

Shell capacity starts at 256 and **grows dynamically** (doubles) as particles are captured.

#### 8.1.5. Snowpack Geometry
```cpp
numParticles      = 1000000;   // total snowpack particle count
spawnLengthS      = 1000.0f;   // patch along slope (m)
spawnWidthZ       = 10.0f;     // patch lateral width (m)
spawnStartS       = 0.0f;      // start offset along slope (m)
spawnThicknessN   = 0.0f;      // 0 = single layer on surface
```

#### 8.1.6. Rigid-Body Drag
```cpp
rollingDrag       = 0.15f;     // rolling friction (linear drag coefficient)
aeroCoeff         = 0.3f;      // aerodynamic drag coefficient (quadratic, scales with R²)
```

#### 8.1.7. Spatial Grid
```cpp
cellSize          = 4 * particleRadius;  // = 0.08 m (default)
```

The grid uses a compact hash table: `hashTableSize = nextPowerOf2(shellCapacity × 8)`, clamped to `[4096, numCells]`. This avoids allocating the full grid (~9.8M cells → ~75 MB) and instead uses ~32K entries (~256 KB) with prime-number spatial hashing (Teschner et al., 2003).

All parameters can be overridden via CLI flags (run with `--help` for the full list).

#### 8.1.8. Performance / Profiling
```cpp
traceInterval     = 0;         // frames between trace snapshots (0 to disable)
logInterval       = 0;         // frames between log outputs (0 = no log)
```

---

## 9. Rendering

### 9.1. Vulkan Backend (optional)

Compiled only with `-DENABLE_VULKAN=ON`. Renders shell particles in real-time.

**Features**:
- CUDA↔Vulkan zero-copy interop via external memory
- GLFW window with orbit camera (mouse drag)
- Ball tracking mode (press T)
- Pre-compiled SPIR-V shaders
- Post-simulation idle loop: edit parameters in a GUI panel and press START to restart

**Requirements**: Vulkan SDK, GLFW 3.x. On Windows, pre-compiled GLFW at `C:\glfw-3.4`. On Linux, `apt install libglfw3-dev`.

---

## 10. References

- NVIDIA CUDA Toolkit docs: https://docs.nvidia.com/cuda/
- NVIDIA Nsight Systems & Compute: https://developer.nvidia.com/tools-overview
- CUB Library: https://nvidia.github.io/cccl/unstable/cub/index.html
- Vulkan API Specification: https://vulkan.lunarg.com/sdk/home
- Dear Imgui API Reference: https://github.com/ocornut/imgui
- Cundall & Strack (1979): A discrete numerical model for granular assemblies.
- Teschner et al. (2003): Optimized Spatial Hashing for Collision Detection of Deformable Objects.
- Pöschel & Schwager (2005): Computational Granular Dynamics.
- Green (2010): Particle Simulation using CUDA.

---