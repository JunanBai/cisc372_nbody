#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define n NUMENTITIES
#define BLOCK_SIZE 256

// CUDA kernel for calculating pairwise accelerations
__global__ void calculate_accelerations(vector3 *dPos, vector3 *dAccels, double *dMass) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i != j) {
            vector3 dist;
            for (int k = 0; k < 3; ++k) {
                dist[k] = dPos[j][k] - dPos[i][k];
            }

            double dist_sq = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];
            double dist_mag = sqrt(dist_sq);
            double accel_mag = GRAV_CONSTANT * dMass[j] / (dist_sq * dist_mag);

            for (int k = 0; k < 3; ++k) {
                dAccels[i*NUMENTITIES+j][k] = accel_mag * dist[k];
            }
        } else {
            for (int k = 0; k < 3; ++k) {
                dAccels[i*NUMENTITIES+j][k] = 0.0;
            }
        }
    }
}

// CUDA kernel for calculating new velocities and positions
__global__ void calculate_velocities_positions(vector3 *dPos, vector3 *dVel, double *dMass) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    vector3 accel_sum = {0, 0, 0};
    vector3 *dAccels;
    if (i < NUMENTITIES) {
        for (int j = 0; j < NUMENTITIES; ++j) {
            for (int k = 0; k < 3; ++k) {
                accel_sum[k] += dAccels[i*NUMENTITIES+j][k];
            }
        }

        for (int k = 0; k < 3; ++k) {
            dVel[i][k] += accel_sum[k] * INTERVAL;
            dPos[i][k] += dVel[i][k] * INTERVAL;
        }
    }
}

void compute(vector3 *hPos, vector3 *hVel, double *mass){
    // Allocate memory on device
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    vector3 *dPos, *dVel, *dAccels;
    double *dMass;
    size_t size = NUMENTITIES * sizeof(vector3);
    size_t sizeMass = NUMENTITIES * sizeof(double);
    size_t sizeAccels = NUMENTITIES * NUMENTITIES * sizeof(vector3);

    cudaMalloc(&dPos, size);
    cudaMalloc(&dVel, size);
    cudaMalloc(&dMass, sizeMass);
    cudaMalloc(&dAccels, sizeAccels);

    cudaMemcpy(dPos, hPos, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, sizeMass, cudaMemcpyHostToDevice);

    calculate_accelerations<<<numBlocks, threadsPerBlock>>>(dPos, dAccels, dMass);
    cudaDeviceSynchronize();  // Wait for the GPU to finish before accessing on host

    calculate_velocities_positions<<<numBlocks, threadsPerBlock>>>(dPos, dVel, dMass);
    cudaDeviceSynchronize();  

    cudaMemcpy(hPos, dPos, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
    cudaFree(dAccels);
}

