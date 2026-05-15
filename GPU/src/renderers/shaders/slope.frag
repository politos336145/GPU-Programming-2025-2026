#version 450

// ============================================================================
// Angry Santa - Slope Fragment Shader (textured snow surface)
//
// Samples a snow texture and applies subtle lighting.
// The texture is a procedurally-generated snow surface (created once at init)
// or loaded from an image file if available.
//
// Descriptor set 0, binding 0: combined image sampler (snow texture)
// ============================================================================

layout(set = 0, binding = 0) uniform sampler2D snowTexture;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main(void) {
    vec4 texColor = texture(snowTexture, fragUV);

    // Subtle directional shading: slightly brighter at the top of the slope
    // fragUV.y goes 0 (top) → tilingV (bottom); use fract to get local tile position
    float shade = mix(1.05, 0.90, fract(fragUV.y));
    // float shade = mix(1.05, 0.90, fragUV.y);

    outColor = vec4(texColor.rgb * shade, 0.92);
}
