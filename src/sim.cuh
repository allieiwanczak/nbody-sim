#pragma once

#include <cuda_runtime.h>

// constants
constexpr int TILE_SIZE = 256;
constexpr float G = 1;
constexpr float SOFTENING = 0.1f;
constexpr float DT = 0.01f;

struct Bodies {
    float *px, *py, *pz;
    float *vx, *vy, *vz;
    float *mass;
    int n;
};

struct Derivatives { 
    float *dpx, *dpy, *dpz;
    float *dvx, *dvy, *dvz;
};

// kernel declarations
__global__ void forceKernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const float* __restrict__ mass,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    float* __restrict__ dpx,
    float* __restrict__ dpy,
    float* __restrict__ dpz,
    float* __restrict__ dvx,
    float* __restrict__ dvy,
    float* __restrict__ dvz,
    int n
);

__global__ void integrateKernel(
    Bodies bodies,
    const Derivatives k1,
    const Derivatives k2,
    const Derivatives k3,
    const Derivatives k4,
    float dt
);

// helpers
void allocBodies(Bodies& b, int n);
void freeBodies(Bodies& b);
void allocDerivatives(Derivatives& d, int n);
void freeDerivatives(Derivatives& d);

void interleavePositions(const float* px, const float* py, const float* pz,
                         float* out, int n);

// RK4
struct TempState { 
    float *px, *py, *pz;
    float *vx, *vy, *vz;
};

__global__ void advanceStateKernel(
    const float* __restrict__ bpx, const float* __restrict__ bpy, const float* __restrict__ bpz,
    const float* __restrict__ bvx, const float* __restrict__ bvy, const float* __restrict__ bvz,
    const float* __restrict__ dpx, const float* __restrict__ dpy, const float* __restrict__ dpz,
    const float* __restrict__ dvx, const float* __restrict__ dvy, const float* __restrict__ dvz,
    float* __restrict__ opx, float* __restrict__ opy, float* __restrict__ opz,
    float* __restrict__ ovx, float* __restrict__ ovy, float* __restrict__ ovz,
    float scale, int n
);

void simInit(int n);
void simFree();
void stepRK4(Bodies& b, float dt);
