#version 460 core

in float vSpeed;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = dot(coord, coord);
    if (r>0.25) discard;

    float brightness = 1.0 - (r/0.25);
    brightness = pow(brightness, 2.0);

    // normalize speed
    float vMax = 1;
    float t = clamp(vSpeed / vMax, 0.0, 1.0);

    // colors by speed
    vec3 slow   = vec3(0.1, 0.3, 1.0);  
    vec3 medium = vec3(1.0, 0.9, 0.3);
    vec3 fast   = vec3(1.0, 0.3, 0.05);

    vec3 color = t < 0.5
        ? mix(slow, medium, t * 2.0)
        : mix(medium, fast, (t - 0.5) * 2.0);

    fragColor = vec4(color, brightness);
}