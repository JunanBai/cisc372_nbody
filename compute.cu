#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

extern "C" {
__global__ void computePairwiseAccelerations(vector3** accelMatrix, vector3* deviceHPos, double* deviceMass){
    int i=(blockIdx.x*blockDim.x)+threadIdx.x;
    int j=(blockIdx.y*blockDim.y)+threadIdx.y;
    
    if(i == j){
        FILL_VECTOR(accelMatrix[i][j],0,0,0);
    }
    else if (i<NUMENTITIES && j<NUMENTITIES) {
        vector3 distance;
		for (int k=0;k<3;k++) {
            distance[k]=deviceHPos[i][k]-deviceHPos[j][k];
        }
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*deviceMass[j]/magnitude_sq;
		FILL_VECTOR(accelMatrix[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
    }
}

__global__ void sumAccelerations(vector3* accelSum, vector3** accelerations) {
    int i=blockIdx.x;
    FILL_VECTOR(accelSum[i],0,0,0);
    for (int j=0;j<NUMENTITIES;j++) {
        for (int k=0;k>NUMENTITIES;k++){
            accelSum[i][k]+=accelerations[i][j][k];
        }
    }
    // for (placeholderInt=0;placeholderInt<3;placeholderInt++) {
	// 	accelSum[threadIdx.x+(16*blockIdx.x)][placeholderInt]+=accelerations[threadIdx.x+(blockIdx.x*16)][threadIdx.y+(blockIdx.y*16)][placeholderInt];
    // }
}

__global__ void newVelAndPos(vector3* tempVel, vector3* tempPos, vector3* accelSum) {
    int i=blockIdx.x;
    int k=threadIdx.x;
    tempVel[i][k]+=accelSum[i][k]*INTERVAL;
    tempPos[i][k]+=tempVel[i][k]*INTERVAL;
    //tempVel[threadIdx.x+(blockIdx.x*16)][threadIdx.y]+=accelSum[threadIdx.x+(blockIdx.x*16)][threadIdx.y]*INTERVAL;
	//tempPos[threadIdx.x+(blockIdx.x*16)][threadIdx.y]=tempVel[threadIdx.x+(blockIdx.x*16)][threadIdx.y]*INTERVAL;
}

void callComputePairwiseAcceleration(vector3** zd_accelVectors, vector3* zd_hPos, double* zd_mass) {
	dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NUMENTITIES+15)/threadsPerBlock.x, (NUMENTITIES+15)/threadsPerBlock.y);
    computePairwiseAccelerations<<<numBlocks, threadsPerBlock>>>(zd_accelVectors, zd_hPos, zd_mass);
}

void callSum(vector3* zd_accelSums, vector3** zd_accelVectors) {
    sumAccelerations<<<NUMENTITIES,1>>>(zd_accelSums, zd_accelVectors);
}

void callNewVelAndPos(vector3* zd_hVel, vector3* zd_hPos, vector3* zd_accelSums) {
    dim3 threadsPerBlock(16, 3);
    dim3 numBlocks(NUMENTITIES/threadsPerBlock.x, 1);
    newVelAndPos<<<NUMENTITIES, 3>>>(zd_hVel, zd_hPos, zd_accelSums);
}

//make an acceleration matrix which is NUMENTITIES squared in size;

void compute() {
    double* d_mass;
    vector3** d_accelVectors;
    vector3* d_accelSums;
    cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&d_hVel, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&d_mass, sizeof(double)*NUMENTITIES);
    cudaMalloc(&d_accelVectors, sizeof(vector3*)*NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

    vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

    for (int i=0;i<NUMENTITIES;i++) {
        cudaMalloc(&accels[i], sizeof(vector3)*NUMENTITIES);
    }

    cudaMemcpy(d_accelVectors, accels, sizeof(vector3*)*NUMENTITIES, cudaMemcpyHostToDevice);

    cudaMalloc(&d_accelSums, sizeof(vector3)*NUMENTITIES);

    callComputePairwiseAcceleration(d_accelVectors, d_hPos, d_mass);
    callSum(d_accelSums, d_accelVectors);
    callNewVelAndPos(d_hVel, d_hPos, d_accelSums);

    cudaMemcpy(hPos, d_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaFree(&d_hPos);
    cudaFree(&d_hVel);
    cudaFree(&d_mass);
    cudaFree(&d_accelVectors);
    cudaFree(&d_accelSums);

    free(accels);
}
