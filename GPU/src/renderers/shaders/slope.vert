#version 450

// ============================================================================
// Angry Santa - Slope Vertex Shader (textured)
//
// Input:  per-vertex position (vec3) + texture coordinates (vec2)
// Output: gl_Position (clip space), fragUV (texture coordinates)
//
// Push constants: mat4 MVP (64 bytes)
// ============================================================================

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    float pointSize;   // unused here, but matches shared push-constant range (68 B)
} pc;

void main(void) {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragUV      = inUV;
}
