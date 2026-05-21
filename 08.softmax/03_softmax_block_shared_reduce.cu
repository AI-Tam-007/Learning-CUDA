#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>


// ================================================================
// CPU Reference
// ================================================================

void softmax_cpu(float* input, float* output, int rows, int cols){

    for(int j = 0; j < rows; j++){

        float total = 0.0f;
        float MAX = -FLT_MAX;

        for(int i = 0; i < cols; i++){

            MAX = fmaxf(input[j * cols + i], MAX);

        }

        for(int i = 0; i < cols; i++){

            total += expf(input[j * cols + i] - MAX);
        }

        for(int i = 0; i < cols; i++){

            output[j * cols + i] = expf(input[j * cols + i] - MAX) / total;

        }

    }
}

void load_float_bin(const char* filename, float* data, int N) {
    std::ifstream fin(filename, std::ios::binary);

    if (!fin) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fin.read(reinterpret_cast<char*>(data), N * sizeof(float));

    if (!fin) {
        std::cerr << "Failed to read enough data from file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fin.close();
}


// ================================================================
// Correctness Check
// ================================================================

void check_cpu(float *output_cpu, float *output_gpu2cpu, int N){

    for(int i = 0; i < N; i++){

        if(fabsf(output_cpu[i] - output_gpu2cpu[i]) > 1e-5f){
            std::cout << "error!" << std::endl;
            return;
        }
    }
    std::cout << "right!" << std::endl;
}

// ================================================================
// CUDA Kernel
// ================================================================

template<int Block_Size>
__global__ void softmax_v1(float *d_input, float *d_output, int rows, int cols){   // 一个线程对应一个元素

    

    int row = blockIdx.x; // 已知第几个block
    int tid = threadIdx.x;

    if(row >= rows){
        return;
    }

    __shared__ float sdata[Block_Size];

    int row_start = row * cols;

    // 求max
    float local_max = -FLT_MAX;

    for(int col = tid; col < cols; col += Block_Size){
        float x = d_input[row_start + col];
        local_max = fmaxf(local_max, x);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // reduce-max-reduce的v3版本了，后面优化：1.展开最后一个warp并消除__syncthreads(); 2.展开for循环;  3.利用warp特性（不一定有提升，要根据数据量分析）
    for(int stride = Block_Size / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    __syncthreads();

    // 求sum
    float local_sum = 0.0f;

    for(int col = tid; col < cols; col += Block_Size){
        float x = d_input[row_start + col];
        local_sum += expf(x - max_val);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for(int stride = Block_Size / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = sdata[0];
    __syncthreads();

    // 求softmax
    for(int col = tid; col < cols; col += Block_Size){
        float x = d_input[row_start + col];
        d_output[row_start + col] = expf(x - max_val) / sum_val;
    }

}



int main(){

    
    int rows = 8192;
    int cols = 4096;

    int N = rows*cols;

    float *input = (float*)malloc(N * sizeof(float));
    float *output = (float*)malloc(N * sizeof(float));
    float *output_gpu2cpu = (float*)malloc(N * sizeof(float));
    

    load_float_bin("data/normal_8192x4096_fp32.bin", input, N);

    softmax_cpu(input, output, rows, cols);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    constexpr int Block_Size = 256;

    if(Block_Size > deviceProp.maxThreadsPerBlock){
        std::cout << "Block_Size exceeds maxThreadsPerBlock!" << std::endl;
        return -1;
    }
    dim3 block(Block_Size);

    int gridSize = std::min<int>(rows, deviceProp.maxGridSize[0]); // 1个block处理一行数据

    dim3 grid(gridSize);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 热身
    softmax_v1<Block_Size><<<grid, block>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    const int iterations = 10; // 迭代10次核函数，计算更精确的gpu运行时间
    float milliseconds = 0; // 计算gpu运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 启动核函数
    for(int i = 0; i < iterations; i++){ // 执行10次，为了计算更精确的时间

        softmax_v1<Block_Size><<<grid, block>>>(d_input, d_output, rows, cols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / iterations;
    std::cout << "reduce_baseline latency = " << avg_ms << " ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){

        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu2cpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_cpu(output, output_gpu2cpu, N);

    free(input);
    free(output);
    free(output_gpu2cpu);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
