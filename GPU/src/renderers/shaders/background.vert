#version 450

// ============================================================================
// Angry Santa - Fullscreen Background Vertex Shader
//
// Draws a fullscreen triangle with no vertex input.
// Uses gl_VertexIndex to generate clip-space positions that cover the screen.
// ============================================================================

layout(location = 0) out vec2 fragUV;

void main(void) {
    // Fullscreen triangle: 3 vertices covering [-1,1] clip space
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    fragUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(fragUV * 2.0 - 1.0, 0.9999, 1.0);  // near max depth

    // Vulkan clip-space Y points downward, so fragUV.y == 0 is the top of
    // the screen. The fragment shader was authored with OpenGL convention
    // (Y == 0 at the bottom). Flip Y so the sky gradient and moon placement
    // remain correct.
    fragUV.y = 1.0 - fragUV.y;
}
