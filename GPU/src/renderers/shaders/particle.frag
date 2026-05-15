#version 450

// ============================================================================
// Angry Santa - Fragment Shader
//
// Handles both point sprites (particles) and filled triangles (slope/ground).
// Uses fragIsPoint flag from vertex shader.
// ============================================================================

layout(location = 0) in  vec3  fragColor;
layout(location = 1) in  float fragIsPoint;
layout(location = 0) out vec4  outColor;

void main(void) {
    if (fragIsPoint > 0.5) {
        // Point sprite: circular with soft edge
        vec2 coord = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(coord, coord);
        if (r2 > 1.0)
            discard;
        float alpha = 1.0 - smoothstep(0.6, 1.0, r2);
        float shade = 1.0 - 0.3 * r2;
        outColor = vec4(fragColor * shade, alpha);
    } else {
        // Triangle geometry: solid fill with slight shading
        outColor = vec4(fragColor, 0.85);
    }
}
