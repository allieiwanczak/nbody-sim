#include <glad/glad.h>
#include<GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cmath>

#include "sim.cuh"
#include "renderer.cuh"
#include "init.cuh"

// config
constexpr int N = 8192;
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr float FOV = 60.f;
constexpr float NEAR = 0.01f;
constexpr float FAR = 1000.f;

// camera
struct Camera { 
    float theta = 0.3f; //azimuth
    float phi = 0.4f; //elevation
    float r= 20.f; 
    bool dragging = false;
    double lastX = 0, lastY = 0;
};

static Camera cam;

static void mat4Perspective(float* m, float fovDeg, float aspect, float near, float far) {
    float f = 1.f / tanf(fovDeg * 0.5f * (float)M_PI / 180.f);

    m[0] = f/aspect; m[1] = 0; m[2] = 0; m[3] = 0; m[4] = 0; m[5] = f;
    m[6] = 0; m[7] = 0; m[8]= 0; m[9] = 0; m[10] = (far+near)/(near-far);
    m[11]=-1 ; m[12] =0; m[13] = 0; m[14] = (2*far*near)/(near-far); m[15] = 0;
}

static void mat4Look(float*m, float ex, float ey, float ez, float cx, float cy, float cz) {
    float fx = cx - ex, fy = cy-ey, fz = cz -ez;
    float fl = sqrtf(fx*fx+fy*fy+fz*fz);
    fx/=fl; fy /= fl; fz /= fl;

    //up (0,1,0)
    float rx = fy*0-fz, ry = fz*0-fx*0, rz = fx-fy*0;
    float rl = sqrtf(rx*rx+ry*ry+rz*rz);
    rx/=rl; ry/=rl; rz/= rl;

    float ux=ry*fz-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy-ry*fx;

    m[0]=rx;  m[1]=ux;  m[2]=-fx; m[3]=0;
    m[4]=ry;  m[5]=uy;  m[6]=-fy; m[7]=0;
    m[8]=rz;  m[9]=uz;  m[10]=-fz;m[11]=0;
    m[12]=-(rx*ex+ry*ey+rz*ez);
    m[13]=-(ux*ex+uy*ey+uz*ez);
    m[14]= (fx*ex+fy*ey+fz*ez);
    m[15]=1;
}

// callbacks
static void mouseButtonCb(GLFWwindow*, int button, int action, int) {
    if (button ==GLFW_MOUSE_BUTTON_LEFT)
        cam.dragging = (action == GLFW_PRESS);
}

static void cursorPosCb(GLFWwindow* w, double x, double y) {
    if (!cam.dragging) { cam.lastX=x; cam.lastY=y; return; }
    float dx = (float)(x - cam.lastX) * 0.005f;
    float dy = (float)(y - cam.lastY) * 0.005f;
    cam.theta += dx;
    cam.phi    = fmaxf(0.05f, fminf((float)M_PI - 0.05f, cam.phi + dy));
    cam.lastX=x; cam.lastY=y;
}

static void scrollCb(GLFWwindow*, double, double dy) {
    cam.r = fmaxf(1.f, cam.r - (float)dy * 0.5f);
}

static void keyCb(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

// main
int main() {
    // glfw
    if (!glfwInit()) {fprintf(stderr, "glfw init failed :(\n)"); return 1;}

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "nbody", nullptr, nullptr);
    if (!window) { fprintf(stderr, "window creation failed :(\n"); glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // uncapped fps

    glfwSetMouseButtonCallback(window, mouseButtonCb);
    glfwSetCursorPosCallback(window,cursorPosCb);
    glfwSetScrollCallback(window,scrollCb);
    glfwSetKeyCallback(window, keyCb);

    // glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    fprintf(stderr, "glad init failed :(\n"); return 1; }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // additive blending
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, WIDTH, HEIGHT);
    

    //sim
    Bodies bodies;
    allocBodies(bodies, N);
    initPlummer(bodies, N);
    simInit(N);

    //renderer
    Renderer renderer;
    rendererInit(renderer, window, N);

    // uniforms
    float proj[16], view[16];
    mat4Perspective(proj, FOV, (float)WIDTH / HEIGHT, NEAR, FAR);

    GLint uView = glGetUniformLocation(renderer.shader, "view");
    GLint uProj = glGetUniformLocation(renderer.shader, "projection");

    glUseProgram(renderer.shader);
    glUniformMatrix4fv(uProj, 1, GL_FALSE, proj);

    // main loops
    double t0 = glfwGetTime();
    int frames = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        stepRK4(bodies, DT);

        // update camera
        float ex = cam.r * sinf(cam.phi) * cosf(cam.theta);
        float ey = cam.r * cosf(cam.phi);
        float ez = cam.r * sinf(cam.phi) * sinf(cam.theta);
        mat4Look(view, ex, ey, ez, 0.f, 0.f, 0.f);

        glUseProgram(renderer.shader);
        glUniformMatrix4fv(uView, 1,GL_FALSE, view);

        // copy pos to vbo
        rendererUpdate(renderer, bodies);

        // draw
        glClearColor(0.02f, 0.02f, 0.05f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        rendererDraw(renderer, N);

        glfwSwapBuffers(window);

        //fps count
        frames++;
        double t1 = glfwGetTime();
        if (t1-t0>=1.0) {
            printf("fps: %d | N: %d\n", frames, N);
            frames = 0;
            t0 = t1;
        }
    }
    // cleanup
    rendererFree(renderer);
    simFree();
    freeBodies(bodies);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
