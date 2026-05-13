#include "sim.cuh"
#include <cassert>
#include <cstdio>

// force kernel
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
) {
    __shared__ float4 sh[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float xi = 0.f, yi = 0.f, zi = 0.f;
    float ax = 0.f, ay = 0.f, az = 0.f;

    if (i<n) {
        xi = px[i];
        yi = py[i];
        zi = pz[i];
    }

    // tiled loop
    int numTiles = (n + TILE_SIZE -1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        int j = tile * TILE_SIZE + threadIdx.x;
        
        if (j<n) {
            sh[threadIdx.x] = { px[j], py[j], pz[j], mass[j]};
        } else {
            sh[threadIdx.x] = { 0.f, 0.f, 0.f, 0.f};
        }

        __syncthreads();

        // accumulate force from every body in this tile
        if (i<n) {
            #pragma unroll 8
            for (int t=0; t< TILE_SIZE; t++) {
                float dx = sh[t].x - xi;
                float dy = sh[t].y - yi;
                float dz = sh[t].z - zi;

                float dist2 = dx*dx + dy*dy +dz*dz + SOFTENING*SOFTENING;

                float invDist = rsqrtf(dist2);
                float invDist3 = invDist * invDist * invDist;

                float f = G * sh[t].w * invDist3; //mass j

                ax += f* dx;
                ay += f* dy;
                az += f*dz;
            }
        }

        __syncthreads();
    }

    // write derivs
    if  (i<n) {
        dpx[i] = vx[i];
        dpy[i] = vy[i];
        dpz[i]= vz[i];

        dvx[i] = ax;
        dvy[i] = ay;
        dvz[i] = az;
    }
}

// integration kernel
__global__ void integrateKernel(
    Bodies bodies,
    const Derivatives k1,
    const Derivatives k2,
    const Derivatives k3,
    const Derivatives k4,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>= bodies.n) return;

    float dt6 = dt / 6.f;

    bodies.px[i] += dt6 * (k1.dpx[i] + 2.f*k2.dpx[i] + 2.f*k3.dpx[i] + k4.dpx[i]);
    bodies.py[i] += dt6 * (k1.dpy[i] + 2.f*k2.dpy[i] + 2.f*k3.dpy[i] + k4.dpy[i]);
    bodies.pz[i] += dt6 * (k1.dpz[i] + 2.f*k2.dpz[i] + 2.f*k3.dpz[i] + k4.dpz[i]);

    bodies.vx[i] += dt6 * (k1.dvx[i] + 2.f*k2.dvx[i] + 2.f*k3.dvx[i] + k4.dvx[i]);
    bodies.vy[i] += dt6 * (k1.dvy[i] + 2.f*k2.dvy[i] + 2.f*k3.dvy[i] + k4.dvy[i]);
    bodies.vz[i] += dt6 * (k1.dvz[i] + 2.f*k2.dvz[i] + 2.f*k3.dvz[i] + k4.dvz[i]);

}

// memory helpers
void allocBodies(Bodies& b, int n) {
    b.n = n;
    cudaMalloc(&b.px,   n * sizeof(float));
    cudaMalloc(&b.py,   n * sizeof(float));
    cudaMalloc(&b.pz,   n * sizeof(float));
    cudaMalloc(&b.vx,   n * sizeof(float));
    cudaMalloc(&b.vy,   n * sizeof(float));
    cudaMalloc(&b.vz,   n * sizeof(float));
    cudaMalloc(&b.mass, n * sizeof(float));
}

void freeBodies(Bodies& b) {
    cudaFree(b.px); cudaFree(b.py); cudaFree(b.pz);
    cudaFree(b.vx); cudaFree(b.vy); cudaFree(b.vz);
    cudaFree(b.mass);
}

void allocDerivatives(Derivatives& d, int n) {
    cudaMalloc(&d.dpx, n * sizeof(float));
    cudaMalloc(&d.dpy, n * sizeof(float));
    cudaMalloc(&d.dpz, n * sizeof(float));
    cudaMalloc(&d.dvx, n * sizeof(float));
    cudaMalloc(&d.dvy, n * sizeof(float));
    cudaMalloc(&d.dvz, n * sizeof(float));
}

void freeDerivatives(Derivatives& d) {
    cudaFree(d.dpx); cudaFree(d.dpy); cudaFree(d.dpz);
    cudaFree(d.dvx); cudaFree(d.dvy); cudaFree(d.dvz);
}