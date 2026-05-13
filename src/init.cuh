#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "sim.cuh"

// helpers
static inline float randf(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

// plummer sphere
static void initPlummer(Bodies& b, int n, float a = 1.f, float totalMass = 1.f) {
    srand((unsigned)time(nullptr));

    // staging buffers
    float* hpx = new float[n]; float* hpy = new float[n]; float* hpz = new float[n];
    float* hvx = new float[n]; float* hvy = new float[n]; float* hvz = new float[n];
    float* hmass = new float[n];

    float bodyMass = totalMass / (float)n;

    for (int i = 0; i < n; i++) {
        hmass[i] = bodyMass;

        // inverse cdf sampling
        float u = randf(0.f, 1.f);
        float r = a / sqrtf(powf(u, -2.f/3.f) - 1.f);

        // point on sphere
        float cosTheta = randf(-1.f, 1.f);
        float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        float phi =  randf(0.f, 2.f * (float)M_PI);

        hpx[i] = r * sinTheta * cosf(phi);
        hpy[i] = r * sinTheta * sinf(phi);
        hpz[i] = r * cosTheta;

        // velocity rejection sampling
        float vesc = sqrtf(2.f) * powf(1.f + (r*r) / (a*a), -0.25f);

        float q, g;
        do {
            q = randf(0.f, 1.f);
            g = randf(0.f, 01.f);
        } while (g > q*q * powf(1.f - q*q, 3.5f)); // Aarseth condition

        float vmag = q *vesc;

        float cosThV = randf(-1.f, 1.f);
        float sinThV = sqrtf(fmax(0.f,1.f - cosThV * cosThV));
        float phiV = randf(0.f,2.f * (float)M_PI);

        hvx[i] = vmag * sinThV * cosf(phiV);
        hvy[i] = vmag * sinThV * sinf(phiV);
        hvz[i] = vmag * cosThV;
    }

    // subtract center of mass drift
    float cvx = 0.f, cvy = 0.f, cvz = 0.f;
    for (int i =0; i<n; i++) { cvx+= hvx[i]; cvy += hvy[i]; cvz += hvz[i]; }
    cvx /= n; cvy /= n; cvz /=n;
    for (int i =0; i<n; i++) { hvx[i] -= cvx; hvy[i] -= cvy; hvz[i] -= cvz; }

    // upload
    cudaMemcpy(b.px,   hpx,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.py,   hpy,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.pz,   hpz,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vx,   hvx,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vy,   hvy,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vz,   hvz,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.mass, hmass, n*sizeof(float), cudaMemcpyHostToDevice);

    delete[] hpx; delete[] hpy; delete[] hpz;
    delete[] hvx; delete[] hvy; delete[] hvz;
    delete[] hmass;
}

// two galaxy collision

static void initGallaxyCollision(Bodies&b, int n, float a = 1.f) {
    srand((unsigned)time(nullptr));

    int half = n/2;

    float* hpx = new float[n]; float* hpy = new float[n]; float* hpz = new float[n];
    float* hvx = new float[n]; float* hvy = new float[n]; float* hvz = new float[n];
    float* hmass = new float[n];

    float bodyMass = 1.f / (float)n;

    auto plummerBody = [&](int i, float ox, float oy, float oz,
                                float ovx, float ovy, float ovz) {
        hmass[i] = bodyMass;

        float u = randf(0.001f, 0.999f);
        float r  = a/ sqrtf(powf(u, -2.f/ 3.f) -1.f);

        float cosT = randf(-1.f, 1.f);
        float sinT = sqrtf(fmaxf(0.f, 1.f - cosT * cosT));
        float phi = randf(0.f, 2.f*(float)M_PI);

        hpx[i] = ox + r * sinT * cosf(phi);
        hpy[i] = ox + r * sinT * sinf(phi);
        hpz[i] = ox + r * cosT;

        float vesc = sqrtf(2.f) * powf(1.f + (r*r) / (a*a), -0.25f);
        float q, g;
        do { q = randf(0.f, 1.f); g = randf(0.f, 0.1f); }
        while (g > q*q * powf(1.f - q*q, 3.5f)); 

        float vmag = q *vesc;

        float cosTv = randf(-1.f, 1.f);
        float sinTv = sqrtf(fmax(0.f,1.f - cosTv * cosTv));
        float phiv = randf(0.f,2.f * (float)M_PI);

        hvx[i] = ovx + vmag * sinTv * cosf(phiv);
        hvy[i] = ovy + vmag * sinTv * sinf(phiv);
        hvz[i] = ovz + vmag * cosTv;
    };

    // galaxy A: offset -5 on x, moving right
    for (int i = 0 ; i< half; i++) plummerBody(i, -5.f, 0.f, 0.f, 0.5f, 0.f, 0.f );
    
    // galaxy B: offset +5 on x, moving left
    for (int i = half; i < n; i++) plummerBody(i, 5.f, 0.f, 0.f, -0.5f, 0.f, 0.f);

    cudaMemcpy(b.px,   hpx,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.py,   hpy,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.pz,   hpz,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vx,   hvx,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vy,   hvy,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.vz,   hvz,   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.mass, hmass, n*sizeof(float), cudaMemcpyHostToDevice);

    delete[] hpx; delete[] hpy; delete[] hpz;
    delete[] hvx; delete[] hvy; delete[] hvz;
    delete[] hmass;
}