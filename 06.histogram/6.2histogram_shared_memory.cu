#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"



template<int blockSize>
__global__ void histogram(int *gpu_input, int *gpu_output, int N)
{
    __shared__ int shared_memory[blockSize];
    int local_block_threadidx = threadIdx.x;
    int global_block_threadidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_block_threadidx < N)
    {
        shared_memory[local_block_threadidx] = 0;
    }
    __syncthreads();

    for(int index = global_block_threadidx; index < N; index += gridDim.x*blockDim.x)
    {
        atomicAdd(&shared_memory[gpu_input[index]], 1);
    }
    __syncthreads();


    atomicAdd(&gpu_output[local_block_threadidx], shared_memory[local_block_threadidx]);

    
}

bool CheckResult(int *cpu_output, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (cpu_output[i] != groudtruth[i]) {
            return false;
        }
    }
    return true;
}

int main(){
    
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    int blockSize = 256;
    dim3 block(blockSize);


    int gridSize = min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);


    int *cpu_input = (int *)malloc(N * sizeof(int));
    int *cpu_output = (int *)malloc(256 * sizeof(int));



    int *gpu_input;
    int *gpu_output;
    cudaMalloc((void **)&gpu_input, 256 * sizeof(int));
    cudaMalloc((void **)&gpu_output, N * sizeof(int));



    for(int i = 0; i < N; i++){
        cpu_input[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }



    cudaMemcpy(gpu_input, cpu_input, N * sizeof(int), cudaMemcpyHostToDevice);
   

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histogram<<<grid, block>>>(gpu_input, gpu_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(cpu_output, gpu_output, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(cpu_output, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%lf ", cpu_output[i]);
        }
        printf("\n");
    }
    printf("histogram latency = %f ms\n", milliseconds);    

    cudaFree(gpu_input);
    cudaFree(gpu_output);
    free(cpu_input);
    free(cpu_output);
}
