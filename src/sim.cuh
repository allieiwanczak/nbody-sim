#pragma once

#include <cuda_runtime.h>

// constants
constexpr int TILE_SIZE = 256;
constexpr float G = 6.67e-11f;
constexpr float SOFTENING = 0.1f;
constexpr float DT = 0.01f;

struct Bodies {
    float *px, *py, *pz;
    float *vx, *vy, *vx;
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
