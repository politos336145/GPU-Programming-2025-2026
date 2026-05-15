#version 450

// ============================================================================
// Billboard Fragment Shader
//
// Samples the billboard texture and discards pixels close to the chroma key
// color (the image background).  Uses a soft edge for natural blending.
//
// Chroma key reference values (linear RGB, adjust in vulkan_renderer_billboard.inl):
//   Santa           : (0.35, 0.58, 0.72)  thresh 0.28  – muted teal sky
//   Village         : (0.06, 0.16, 0.33)  thresh 0.22  – dark night sky
//   Village Destr.  : same as Village
// ============================================================================

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(push_constant) uniform PC {
    mat4  mvp;
    float chromaR;
    float chromaG;
    float chromaB;
    float chromaThresh;
} pc;

void main(void) {
    vec4  c      = texture(tex, fragUV);
    vec3  key    = vec3(pc.chromaR, pc.chromaG, pc.chromaB);
    float dist   = length(c.rgb - key);

    // Hard discard below threshold, soft fade in the fringe band
    if (dist < pc.chromaThresh) discard;
    float alpha  = smoothstep(pc.chromaThresh, pc.chromaThresh + 0.12, dist);

    outColor = vec4(c.rgb, alpha);
}
