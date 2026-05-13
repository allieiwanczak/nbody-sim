#version 460 core

layout ( location = 0) in vec3 pos;
layout (location =1) in float speed;

uniform mat4 view;
uniform mat4 projection;

out float vSpeed;

void main() {
    gl_Position = projection * view *vec4(pos, 1.0);
    vSpeed = speed;

    float dist = length((view * vec4(pos, 1.0)).xyz);
    gl_PointSize = clamp(10.0 /dist, 1.0, 1.0);
}