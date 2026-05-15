#version 450

// ============================================================================
// Angry Santa - Particle Vertex Shader
//
// Input:  per-vertex position (vec3) + colorCode (float)
// Output: gl_Position (clip space), gl_PointSize, fragColor (RGB)
//
// Color code mapping (set by fillVBOKernel):
//   0.0       = seed particle       → green
//   1.0–1.99  = reserved legacy codes, not emitted by the current renderer
//   2.0–2.99  = captured particle   → cyan gradient (wetness encoded: code-2.0)
//   3.0       = slope surface       → white-grey  (triangle geometry)
//   4.0       = ground plane        → brown       (triangle geometry)
//   5.0+      = snowball            → red, size encodes radius
//
// Push constants: mat4 MVP (64 bytes) + float pointSize (4 bytes)
// ============================================================================

layout(location = 0) in vec3  inPosition;
layout(location = 1) in float inColorCode;

layout(location = 0) out vec3  fragColor;
layout(location = 1) out float fragIsPoint;

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    float pointSize;
} pc;

void main(void) {
    gl_Position  = pc.mvp * vec4(inPosition, 1.0);
    
    // Point size: scale with distance for pseudo-perspective
    float dist = max(gl_Position.w, 0.1);
    float basePtSize = pc.pointSize * (5.0 / dist);
    
    if (inColorCode < 0.5) {
        // Seed shell particles - green
        fragColor = vec3(0.2, 0.9, 0.3);
        gl_PointSize = basePtSize;
    } else if (inColorCode < 1.5) {
        // Reserved legacy range - kept visually distinct if fed by stale data
        fragColor = vec3(0.3, 0.3, 0.3);
        gl_PointSize = basePtSize * 0.5;
    } else if (inColorCode < 3.0) {
        // Captured particles - code = 2.0 + wetness [0,1]
        // Wetness gradient: low wetness → pale cyan, high wetness → bright white-blue
        float w = clamp(inColorCode - 2.0, 0.0, 1.0);
        fragColor = mix(vec3(0.05, 0.1, 0.6),    // blue   (dry)
                        vec3(0.5, 0.0, 0.8),     // violet (wet)
                        w);
        gl_PointSize = basePtSize * 1.2;
    } else if (inColorCode < 3.5) {
        // Slope surface - white-grey snow
        fragColor = vec3(0.88, 0.90, 0.95);
        gl_PointSize = 1.0;
    } else if (inColorCode < 4.5) {
        // Ground plane - brown
        fragColor = vec3(0.45, 0.35, 0.22);
        gl_PointSize = 1.0;
    } else {
        // Snowball - bright blue sphere
        fragColor = vec3(0.15, 0.55, 1.0);
        float ballRadius = (inColorCode - 5.0) * 0.1;
        gl_PointSize = clamp(2.5 * ballRadius * 481.0 / dist, 4.0, 400.0);
    }

    // Flag: 1.0 for point sprites (particles + snowball), 0.0 for triangles (slope/ground)
    fragIsPoint = (inColorCode < 3.0 || inColorCode >= 4.5) ? 1.0 : 0.0;
}
