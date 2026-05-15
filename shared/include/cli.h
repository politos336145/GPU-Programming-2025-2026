#ifndef CLI_H
  #define CLI_H

  int argInt(int argc, char** argv, const char* flag, int def);
  float argFloat(int argc, char** argv, const char* flag, float def);
  bool argBool(int argc, char** argv, const char* flag);
#endif // CLI_H