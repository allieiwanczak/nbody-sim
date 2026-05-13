#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "sim.cuh"

struct Renderer { 
    GLuint vao;
    GLuint vbo;
    GLuint shader;
    GLFWwindow* window;

    cudaGraphicsResource* cudaVboResource;

    int n;
};

void rendererInit(Renderer& r, GLFWwindow* window, int n);
void rendererFree(Renderer& r);

// map VBO into cuda
void rendererUpdate(Renderer& r, const Bodies& b);

// draw call
void rendererDraw(Renderer& r, int n);

// shader helpers
GLuint compileShader(const char* path, GLenum type);
GLuint linkProgram(GLuint vert, GLuint frag);