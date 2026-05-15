// ============================================================================
// ImGui - single translation unit for all ImGui sources
// Compiled by MSVC (CXX), NOT by nvcc. This avoids incompatibilities between
// Dear ImGui's C++11/14 code and CUDA's host compiler restrictions.
// ============================================================================

// Silence MSVC warnings in third-party code
#ifdef _MSC_VER
  #pragma warning(push, 0)
#endif

#include "imgui/imgui.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_tables.cpp"
#include "imgui/imgui_widgets.cpp"

// Backends
#define GLFW_INCLUDE_VULKAN
#include "imgui/backends/imgui_impl_glfw.cpp"
#include "imgui/backends/imgui_impl_vulkan.cpp"

#ifdef _MSC_VER
  #pragma warning(pop)
#endif
