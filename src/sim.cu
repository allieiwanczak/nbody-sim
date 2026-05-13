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

// advance state kernel
__global__ void advanceStateKernel(
    const float* __restrict__ bpx, const float* __restrict__ bpy, const float* __restrict__ bpz,
    const float* __restrict__ bvx, const float* __restrict__ bvy, const float* __restrict__ bvz,
    const float* __restrict__ dpx, const float* __restrict__ dpy, const float* __restrict__ dpz,
    const float* __restrict__ dvx, const float* __restrict__ dvy, const float* __restrict__ dvz,
    float* __restrict__ opx, float* __restrict__ opy, float* __restrict__ opz,
    float* __restrict__ ovx, float* __restrict__ ovy, float* __restrict__ ovz,
    float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    opx[i] = bpx[i] + scale * dpx[i];
    opy[i] = bpy[i] + scale * dpy[i];
    opz[i] = bpz[i] + scale * dpz[i];

    ovx[i] = bvx[i] + scale * dvx[i];
    ovy[i] = bvy[i] + scale * dvy[i];
    ovz[i] = bvz[i] + scale * dvz[i];
}

// RK4 State
static TempState s_tmp;
static Derivatives s_k1, s_k2, s_k3, s_k4;

void simInit(int n) {
    cudaMalloc(&s_tmp.px, n*sizeof(float)); cudaMalloc(&s_tmp.py, n*sizeof(float)); cudaMalloc(&s_tmp.pz, n*sizeof(float));
    cudaMalloc(&s_tmp.vx, n*sizeof(float)); cudaMalloc(&s_tmp.vy, n*sizeof(float)); cudaMalloc(&s_tmp.vz, n*sizeof(float));
    allocDerivatives(s_k1, n);
    allocDerivatives(s_k2, n);
    allocDerivatives(s_k3, n);
    allocDerivatives(s_k4, n);
}

void simFree() {
    cudaFree(s_tmp.px); cudaFree(s_tmp.py); cudaFree(s_tmp.pz);
    cudaFree(s_tmp.vx); cudaFree(s_tmp.vy); cudaFree(s_tmp.vz);
    freeDerivatives(s_k1); freeDerivatives(s_k2);
    freeDerivatives(s_k3); freeDerivatives(s_k4);
}

__global__ void interleaveKernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i*4 + 0] = px[i];
    out[i*4 + 1] = py[i];
    out[i*4 + 2] = pz[i];
    out[i*4 + 3] = sqrtf(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
}

void interleavePositions(const float* px, const float* py, const float* pz,
                         const float* vx, const float* vy, const float* vz,
                         float* out, int n) {
    dim3 block(TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE);
    interleaveKernel<<<grid, block>>>(px, py, pz, vx, vy, vz, out, n);
}

void stepRK4(Bodies& b, float dt) {
    int n = b.n;
    dim3 block(TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE);

    // k1
    forceKernel<<<grid, block>>>(
        b.px, b.py, b.pz, b.mass, b.vx, b.vy, b.vz,
        s_k1.dpx, s_k1.dpy, s_k1.dpz, s_k1.dvx, s_k1.dvy, s_k1.dvz, n
    );

    // k2 : eval at pos + dt/2 * k1
    advanceStateKernel<<<grid,block>>>(
        b.px, b.py, b.pz, b.vx, b.vy, b.vz,
        s_k1.dpx, s_k1.dpy, s_k1.dpz, s_k1.dvx, s_k1.dvy, s_k1.dvz,
        s_tmp.px, s_tmp.py, s_tmp.pz, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        dt * 0.5f, n
    );
    forceKernel<<<grid,block>>>(
        s_tmp.px,s_tmp.py, s_tmp.pz, b.mass, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        s_k2.dpx, s_k2.dpy, s_k2.dpz, s_k2.dvx, s_k2.dvy, s_k2.dvz, n
    );

    // k3: eval at pos + dt/2 *k2
    advanceStateKernel<<<grid,block>>>(
        b.px, b.py, b.pz, b.vx, b.vy, b.vz,
        s_k2.dpx, s_k2.dpy, s_k2.dpz, s_k2.dvx, s_k2.dvy, s_k2.dvz,
        s_tmp.px, s_tmp.py, s_tmp.pz, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        dt * 0.5f, n
    );
    forceKernel<<<grid,block>>>(
        s_tmp.px,s_tmp.py, s_tmp.pz, b.mass, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        s_k3.dpx, s_k3.dpy, s_k3.dpz, s_k3.dvx, s_k3.dvy, s_k3.dvz, n
    );

    // k4: eval at pos + dt *k3
    advanceStateKernel<<<grid,block>>>(
        b.px, b.py, b.pz, b.vx, b.vy, b.vz,
        s_k3.dpx, s_k3.dpy, s_k3.dpz, s_k3.dvx, s_k3.dvy, s_k3.dvz,
        s_tmp.px, s_tmp.py, s_tmp.pz, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        dt, n
    );
    forceKernel<<<grid,block>>>(
        s_tmp.px,s_tmp.py, s_tmp.pz, b.mass, s_tmp.vx, s_tmp.vy, s_tmp.vz,
        s_k4.dpx, s_k4.dpy, s_k4.dpz, s_k4.dvx, s_k4.dvy, s_k4.dvz, n
    );

    // combine all
    integrateKernel<<<grid, block>>>(b, s_k1, s_k2, s_k3, s_k4, dt);
}