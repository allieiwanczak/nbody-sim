#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "renderer.cuh"

// load shaders
static std::string readFile(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "failed to open shader: %s\n", path);
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint compileShader(const char* path, GLenum type) {
    std::string src = readFile(path);
    const char* cstr = src.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &cstr, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        fprintf(stderr, "shader compile error (%s):\n%s\n", path, log);
        exit(1);
    }
    return shader;
}

GLuint linkProgram(GLuint vert, GLuint frag) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        fprintf(stderr, "program link error:\n%s\n", log);
        exit(1);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

// init
void rendererInit(Renderer& r, GLFWwindow* window, int n) {
    r.window = window;
    r.n = n;

    // vao, vbo
    glGenVertexArrays(1, &r.vao);
    glGenBuffers(1, &r.vbo);

    glBindVertexArray(r.vao);
    glBindBuffer(GL_ARRAY_BUFFER, r.vbo);

    // allocate vbo
    glBufferData(GL_ARRAY_BUFFER, n * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // register vbo with cuda
    cudaGraphicsGLRegisterBuffer(
        &r.cudaVboResource,
        r.vbo,
        cudaGraphicsMapFlagsWriteDiscard
    );

    // shaders
    GLuint vert = compileShader("shaders/particles.vert", GL_VERTEX_SHADER);
    GLuint frag = compileShader("shaders/particles.frag", GL_FRAGMENT_SHADER);
    r.shader = linkProgram(vert, frag);
}

// copy pos into vbo
void rendererUpdate(Renderer& r, const Bodies& b) {
    float* dptr = nullptr;
    size_t size = 0;

    cudaGraphicsMapResources(1, &r.cudaVboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, r.cudaVboResource);
    interleavePositions(b.px, b.py, b.pz, dptr, b.n);

    cudaGraphicsUnmapResources(1, &r.cudaVboResource, 0);
}

// draw
void rendererDraw(Renderer& r, int n) {
    glUseProgram(r.shader);
    glBindVertexArray(r.vao);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, n);

    glBindVertexArray(0);
}

// free
void rendererFree(Renderer& r) {
    cudaGraphicsUnregisterResource(r.cudaVboResource);
    glDeleteBuffers(1, &r.vbo);
    glDeleteVertexArrays(1, &r.vao);
    glDeleteProgram(r.shader);
}