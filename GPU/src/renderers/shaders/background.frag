#version 450

// ============================================================================
// Angry Santa - Procedural Night Sky Fragment Shader
//
// Reproduces a stylised night sky: deep blue gradient, scattered four-pointed
// stars of varying size and brightness, and a soft glowing moon.
//
// Designed to match the reference artwork: warm moonlight at upper-left,
// blue-teal gradient darkening toward bottom-right, small diamond-shaped
// sparkle stars.
// ============================================================================

layout(location = 0) in  vec2 fragUV;      // [0,1] screen coordinates
layout(location = 0) out vec4 outColor;

// ---- Pseudo-random hash (good spatial distribution) ----
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    float n = hash21(p);
    return vec2(n, hash21(p + n));
}

void main(void) {
    vec2 uv = fragUV;

    // ---- Sky gradient ----
    // Deep blue at top, slightly lighter teal at bottom (like reference)
    vec3 topColor    = vec3(0.06, 0.18, 0.38);   // dark navy
    vec3 bottomColor = vec3(0.10, 0.28, 0.48);   // slightly brighter blue
    vec3 sky = mix(bottomColor, topColor, uv.y);

    // Subtle radial darkening from moon position
    float moonDist = length(uv - vec2(0.22, 0.82));
    sky += vec3(0.03, 0.04, 0.06) * smoothstep(0.8, 0.0, moonDist);

    // ---- Stars ----
    vec3 starColor = vec3(0.0);
    float starLayer = 0.0;

    // Multiple grid layers for varied star density
    for (int layer = 0; layer < 3; layer++) {
        float scale = 18.0 + float(layer) * 12.0;  // grid cell sizes: 18, 30, 42
        vec2 gv = fract(uv * scale) - 0.5;
        vec2 id = floor(uv * scale);

        vec2 rnd = hash22(id + float(layer) * 100.0);

        // Only ~30% of cells have a star
        if (rnd.x > 0.30) continue;

        // Jitter position within cell
        vec2 starPos = (rnd - 0.5) * 0.7;
        vec2 d = gv - starPos;

        // Four-pointed star shape (diamond sparkle)
        float ax = abs(d.x);
        float ay = abs(d.y);
        float cross4 = min(ax, ay);             // diamond cross
        float pointLen = 0.015 + rnd.y * 0.012; // spike length varies

        // Core glow (circular)
        float coreR = 0.003 + rnd.y * 0.004;
        float core = smoothstep(coreR, coreR * 0.3, length(d));

        // Spikes along axes
        float spikeX = smoothstep(pointLen, 0.0, ax) * smoothstep(0.003, 0.0, ay);
        float spikeY = smoothstep(pointLen, 0.0, ay) * smoothstep(0.003, 0.0, ax);
        float spikes = (spikeX + spikeY) * 0.6;

        // Twinkle: vary brightness subtly per star
        float brightness = 0.5 + 0.5 * rnd.y;

        float star = (core + spikes) * brightness;
        // Warm-white tint like reference image
        starColor += star * vec3(0.95, 0.92, 0.80);
    }

    // ---- Moon ----
    vec2 moonCenter = vec2(0.22, 0.82);   // upper-left area
    float moonR = 0.04;
    float md = length(uv - moonCenter);

    // Solid moon disc
    float moonDisc = smoothstep(moonR, moonR - 0.004, md);

    // Soft glow halo
    float moonGlow = smoothstep(moonR * 4.0, moonR * 0.5, md) * 0.15;

    // Moon colour: warm cream/ivory
    vec3 moonCol = vec3(0.96, 0.94, 0.82);
    // Subtle surface variation
    float moonNoise = hash21(uv * 200.0) * 0.06;
    moonCol -= moonNoise * moonDisc;

    vec3 color = sky + starColor;
    color = mix(color, moonCol, moonDisc);      // moon solid
    color += moonGlow * vec3(0.90, 0.88, 0.70); // soft halo

    outColor = vec4(color, 1.0);
}
