#version 450

// ============================================================================
// Billboard Vertex Shader
//
// Receives corners pre-computed on CPU (camera-facing quad).
// Simply transforms position to clip space and passes UV through.
// Push constants: mat4 MVP + chroma key params (shared with fragment shader).
// ============================================================================

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform PC {
    mat4  mvp;
    float chromaR;
    float chromaG;
    float chromaB;
    float chromaThresh;
} pc;

void main(void) {
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    fragUV      = inUV;
}
