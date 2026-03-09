// CUDA直方图统计数组内元素出现频次 



#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
//2.386ms
__global__ void histogram(int *gpu_input, int *gpu_output)
{
    int global_block_threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    /*error*/
    // bin_data[hist_data[gtid]]++;
    /*right*/
    // 原子加法，将所有并行线程强制转为串行，但不保证顺序
    atomicAdd(&gpu_output[gpu_input[global_block_threadidx]], 1);
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
