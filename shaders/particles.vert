#version 460 core

layout ( location = 0) in vec3 pos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view *vec4(pos, 1.0);

    float dist = length((view * vec4(pos, 1.0)).xyz);
    gl_PointSize = clamp(80/0 /dist, 1.0, 8.0);
}