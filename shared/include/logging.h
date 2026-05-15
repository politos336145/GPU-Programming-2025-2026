#ifndef LOGGING_H
  #define LOGGING_H

  #include <cstdio>
  
  struct FrameStats {
    int frame;
    int shellCount;            // current shell particle count
    int snowpackAlive;         // remaining alive snowpack particles
    float ballMass;
    float ballRadius;
    float ballPosX, ballPosY, ballPosZ;
    float ballVelX, ballVelY, ballVelZ;
    float frameTimeMs;         // total frame time (cudaEvent)
    float captureMs;           // snowpack -> shell capture
    float shellForcesTetherMs; // forcesTether: gravity + damping + tether (fused)
    float shellGridMs;         // spatial grid build
    float shellCollisionMs;    // neighbor collision + cohesion
    float shellIntGroundMs;    // intGround: integration + ground collision (fused)
    float coreUpdateMs;        // rigid body update
  };

  class Logger {
    private:
      FILE *_csvFile;

    public:
      Logger(void);
      ~Logger(void);

      bool openCSV(const char *filename);
      void writeHeader(void);
      void logFrame(const FrameStats &stats);
      void close(void);
  };
#endif // LOGGING_H