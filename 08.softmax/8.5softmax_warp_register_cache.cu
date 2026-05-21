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
// Configuration
// ================================================================

#define WARP_SIZE 32

// ================================================================
// CPU Reference
// ================================================================

void softmax_cpu(float* input, float* output, int rows, int cols) {

    for (int j = 0; j < rows; j++) {

        float total = 0.0f;
        float MAX = -FLT_MAX;

        for (int i = 0; i < cols; i++) {
            MAX = fmaxf(input[j * cols + i], MAX);
        }

        for (int i = 0; i < cols; i++) {
            total += expf(input[j * cols + i] - MAX);
        }

        for (int i = 0; i < cols; i++) {
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

void check_cpu(float* output_cpu, float* output_gpu2cpu, int N) {

    for (int i = 0; i < N; i++) {

        if (fabsf(output_cpu[i] - output_gpu2cpu[i]) > 1e-5f) {
            std::cout << "error" << std::endl;
            return;
        }
    }

    std::cout << "right!" << std::endl;
}

// ================================================================
// CUDA Device Helpers
// ================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }

    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

// ================================================================
// CUDA Kernel
// ================================================================

template<int Cols_Per_Thread>
__global__ void softmax_v3_warp_per_row_register_cache(
    float* d_input,
    float* d_output,
    int rows, 
    int cols
) {
    int lane = threadIdx.x; // 0 ~ 31
    int warp_in_block = threadIdx.y; // block 内第几个 warp

    int row = blockIdx.x * blockDim.y + warp_in_block;

    if (row >= rows) {
        return;
    }

    int row_start = row * cols;

    // 每个线程缓存自己负责的 Cols_Per_Thread 个元素
    float buf[Cols_Per_Thread];

    // step 1: 从 global memory 读取到寄存器 buf，同时求 local_max
    float local_max = -FLT_MAX;

#pragma unroll
    for (int i = 0; i < Cols_Per_Thread; i++) {

        int col = i * WARP_SIZE + lane;

        if (col < cols) {
            float x = d_input[row_start + col];
            buf[i] = x;
            local_max = fmaxf(local_max, x);
        } else {
        buf[i] = -FLT_MAX;
    }
}

// step 2: warp 内 reduce max
float max_val = warp_reduce_max(local_max);

// 因为 __shfl_down_sync reduce 结果只保证在 lane 0，
// 所以要从 lane 0 广播给整个 warp
max_val = __shfl_sync(0xffffffff, max_val, 0);

// step 3: 直接使用寄存器 buf 计算 expf(x - max)，并求 local_sum
float local_sum = 0.0f;

#pragma unroll
for (int i = 0; i < Cols_Per_Thread; i++) {

    int col = i * WARP_SIZE + lane;

    if (col < cols) {
        float e = expf(buf[i] - max_val);
        buf[i] = e;
        local_sum += e;
    } else {
    buf[i] = 0.0f;
}
}

// step 4: warp 内 reduce sum
float sum_val = warp_reduce_sum(local_sum);

// 同样从 lane 0 广播给整个 warp
sum_val = __shfl_sync(0xffffffff, sum_val, 0);

// step 5: 复用 buf 中的 expf(x - max)，除以 sum 后写回
#pragma unroll
for (int i = 0; i < Cols_Per_Thread; i++) {

    int col = i * WARP_SIZE + lane;

    if (col < cols) {
        d_output[row_start + col] = buf[i] / sum_val;
    }
}
}



int main() {

    int rows = 4096;
    int cols = 64;
    int N = rows * cols;

    float* input = (float*)malloc(N * sizeof(float));
    float* output_cpu = (float*)malloc(N * sizeof(float));
    float* output_gpu2cpu = (float*)malloc(N * sizeof(float));

    load_float_bin("data/normal_4096x64_fp32.bin", input, N);

    softmax_cpu(input, output_cpu, rows, cols);

    float* d_input;
    float* d_output;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(32, 8);
    dim3 grid((rows + block.y - 1) / block.y);

    // 当前 cols = 1024，一个 warp 32 个线程
    // 每个线程处理 1024 / 32 = 32 个元素
    constexpr int Cols_Per_Thread = 2;

    // warmup
    softmax_v3_warp_per_row_register_cache<Cols_Per_Thread><<<grid, block>>>(d_input, d_output, rows, cols);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Warmup kernel launch error: "
        << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    const int iterations = 10;
    float milliseconds = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        softmax_v3_warp_per_row_register_cache<Cols_Per_Thread>
        <<<grid, block>>>(d_input, d_output, rows, cols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    float avg_ms = milliseconds / iterations;

    std::cout << "softmax_v3_warp_per_row_register_cache latency = "
    << avg_ms << " ms" << std::endl;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: "
        << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu2cpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    check_cpu(output_cpu, output_gpu2cpu, N);

    std::cout << "cpu[0] = " << output_cpu[0] << std::endl;
    std::cout << "gpu[0] = " << output_gpu2cpu[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    free(input);
    free(output_cpu);
    free(output_gpu2cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
