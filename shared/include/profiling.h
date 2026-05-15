#ifndef PROFILING_H
  #define PROFILING_H

  struct ProfilingStats {
    static constexpr int NUM_KERNELS = 7;
    const char* kernelNames[NUM_KERNELS] = { "capture", "forcesTether", "shellGrid", "shellCollision", "intGround", "shellFused", "coreUpdate" };
    double totalMs[NUM_KERNELS];
    float  minMs[NUM_KERNELS];
    float  maxMs[NUM_KERNELS];
    double totalFrameMs;
    float  minFrameMs;
    float  maxFrameMs;
    int    frameCount;

    /**
     * @brief Reset all stats to initial state.
     */
    void reset(void) {
      for (int i = 0; i < NUM_KERNELS; i++) {
        totalMs[i] = 0.0;
        minMs[i]   = 1e9f;
        maxMs[i]   = 0.0f;
      }

      totalFrameMs = 0.0;
      minFrameMs   = 1e9f;
      maxFrameMs   = 0.0f;
      frameCount   = 0;
    }

    /**
     * @brief Accumulate kernel timings from a single frame.
     * @param kt  KernelTimings struct with per-kernel elapsed times.
     */
    void accumulate(const KernelTimings& kt) {
      float vals[NUM_KERNELS] = { kt.captureMs, kt.shellForcesMs, kt.shellGridMs, kt.shellCollisionMs, kt.shellIntegrateMs, kt.shellFusedMs, kt.coreUpdateMs };

      for (int i = 0; i < NUM_KERNELS; i++) {
        totalMs[i] += vals[i];
        if (vals[i] < minMs[i]) minMs[i] = vals[i];
        if (vals[i] > maxMs[i]) maxMs[i] = vals[i];
      }

      totalFrameMs += kt.totalFrameMs;
      if (kt.totalFrameMs < minFrameMs) minFrameMs = kt.totalFrameMs;
      if (kt.totalFrameMs > maxFrameMs) maxFrameMs = kt.totalFrameMs;
      frameCount++;
    }

    /**
     * @brief Print a summary report of profiling results to stdout.
     * @param shellN      Final shell particle count.
     * @param snowpackN   Total snowpack particle count.
     * @param totalFrames Total number of frames simulated (optional, for display).
     */
    void printSummary(int shellN, int snowpackN, int totalFrames = -1) const {
      printf("======================================================================\n");
      if (totalFrames > 0 && totalFrames != frameCount)
        printf("  PROFILING SUMMARY  (shell=%d, snowpack=%d, %d frames)\n", shellN, snowpackN, totalFrames);
      else
        printf("  PROFILING SUMMARY  (shell=%d, snowpack=%d, %d frames)\n", shellN, snowpackN, frameCount);
      printf("======================================================================\n");
      printf("  %-24s  %10s  %10s  %10s\n", "Kernel", "Avg (ms)", "Min (ms)", "Max (ms)");
      printf("  %-24s  %10s  %10s  %10s\n", "------------------------", "----------", "----------", "----------");
      
      double sumAvg = 0.0;
      for (int i = 0; i < NUM_KERNELS; i++) {
        double avg = (frameCount > 0) ? totalMs[i] / frameCount : 0.0;
        sumAvg += avg;
        printf("  %-24s  %10.4f  %10.4f  %10.4f\n", kernelNames[i], avg, minMs[i], maxMs[i]);
      }

      printf("  %-24s  %10s  %10s  %10s\n", "------------------------", "----------", "----------", "----------");
      printf("  %-24s  %10.4f\n", "Sum kernels", sumAvg);
      
      double avgFrame = (frameCount > 0) ? totalFrameMs / frameCount : 0.0;
      printf("  %-24s  %10.4f  %10.4f  %10.4f\n", "Total frame", avgFrame, minFrameMs, maxFrameMs);
      
      double overhead = avgFrame - sumAvg;
      printf("  %-24s  %10.4f\n", "Overhead (memcpy etc.)", overhead);
      printf("----------------------------------------------------------------------\n");
      
      double fps = (avgFrame > 0.0) ? 1000.0 / avgFrame : 0.0;
      printf("  Effective FPS:          %.1f\n", fps);
      
      int maxIdx = 0;
      for (int i = 1; i < NUM_KERNELS; i++) {
        if (totalMs[i] > totalMs[maxIdx]) maxIdx = i;
      }
      
      double pct = (sumAvg > 0.0) ? 100.0 * (totalMs[maxIdx] / frameCount) / sumAvg : 0.0;
      printf("  Bottleneck:             %s (%.1f%%)\n", kernelNames[maxIdx], pct);
      printf("======================================================================\n");
    }
  };
#endif // PROFILING_H