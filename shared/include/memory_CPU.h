#ifndef MEMORY_CPU_H
  #define MEMORY_CPU_H

  #ifdef _MSC_VER
    #include <malloc.h>
  #endif

  // Shared allocation macros for CPU code paths.
  // In SIMD builds, allocate 16-byte aligned memory for packed loads/stores.
  #ifdef SIMD
    #ifdef _MSC_VER
      #define ALLOC(bytes)  _aligned_malloc((bytes), 16)
      #define FREE(ptr)     _aligned_free(ptr)
    #else
      static inline void* simd_alloc_impl_(size_t bytes) {
        size_t aligned = (bytes + 15u) & ~(size_t)15u;
        return aligned_alloc(16, aligned > 0 ? aligned : 16);
      }
      #define ALLOC(bytes)  simd_alloc_impl_(bytes)
      #define FREE(ptr)     free(ptr)
    #endif
  #else
    #define ALLOC(bytes)  malloc(bytes)
    #define FREE(ptr)     free(ptr)
  #endif
#endif // MEMORY_CPU_H
