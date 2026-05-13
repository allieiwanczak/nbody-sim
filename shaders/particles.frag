#version 460 core

out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = dot(coord, coord);
    if (r>0.25) discard;

    float brightness = 1.0 - (r/0.25);
    brightness = pow(brightness, 2.0);

    // warm white core, blueish fringe
    vec3 color = mix(vec3(0.4,0.6,1.0), vec3(1.0, 0.95, 0.8), brightness);

    fragColor = vec4(color, brightness);
}