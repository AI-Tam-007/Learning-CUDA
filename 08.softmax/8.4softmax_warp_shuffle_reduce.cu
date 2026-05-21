

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
            std::cout << "error at " << i
            << ", cpu = " << output_cpu[i]
            << ", gpu = " << output_gpu2cpu[i]
            << std::endl;
            return;
        }
    }

    std::cout << "right!" << std::endl;
}

// ================================================================
// CUDA Device Helpers
// ================================================================

__device__ __forceinline__ float warp_reduce_max(float val){      // shfl_down当前warp线程往后偏移offset位获取val

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);   // 这个意思就是0和16对比，1和17对比，……15和31对比，然后继续规约
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

__global__ void softmax_v2_warp_per_row(
    float* d_input,
    float* d_output,
    int rows, 
    int cols
) {
    int lane = threadIdx.x; // 当前线程在 warp 内的编号: 0 ~ 31
    int warp_in_block = threadIdx.y; // 当前 warp 是 block 内第几个 warp：0 ~ 7

    int row = blockIdx.x * blockDim.y + warp_in_block;  // 全局warp

    if (row >= rows){
        return;
    }

    int row_start = row * cols;

    // step 1: 当前 lane 负责这一行中的若干列，先求 local max
    float local_max = -FLT_MAX;

    for (int col = lane; col < cols; col += WARP_SIZE) {
        float x = d_input[row_start + col];
        local_max = fmaxf(local_max, x);
    }

    // step 2: warp 内 reduce，得到这一行的 max
    float max_val = warp_reduce_max(local_max);

    // 注意：
    // __shfl_down_sync 规约后的最终完整结果只保证在 lane 0 上。
    // 所以这里需要把 lane 0 的 max_val 广播给整个 warp。
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // step 3: 当前 lane 负责这一行中的若干列，求 local sum
    float local_sum = 0.0f;

    for (int col = lane; col < cols; col += WARP_SIZE) {
        float x = d_input[row_start + col];
        local_sum += expf(x - max_val);
    }

    // step 4: warp 内 reduce，得到这一行的 sum
    float sum_val = warp_reduce_sum(local_sum);

    // 同理，把 lane 0 的 sum_val 广播给整个 warp
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);

    // step 5: 当前 lane 负责这一行中的若干列，写回 softmax 结果
    for (int col = lane; col < cols; col += WARP_SIZE) {
        float x = d_input[row_start + col];
        d_output[row_start + col] = expf(x - max_val) / sum_val;
    }
}


int main() {

    int rows = 128;
    int cols = 256;
    int N = rows * cols;

    float* input = (float*)malloc(N * sizeof(float));
    float* output_cpu = (float*)malloc(N * sizeof(float));
    float* output_gpu2cpu = (float*)malloc(N * sizeof(float));

    load_float_bin("data/normal_128x256_fp32.bin", input, N);
    softmax_cpu(input, output_cpu, rows, cols);

    float* d_input;
    float* d_output;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    // 每个 warp 处理一行
    dim3 block(32, 8);
    dim3 grid((rows + block.y - 1) / block.y);

    // warmup
    softmax_v2_warp_per_row<<<grid, block>>>(d_input, d_output, rows, cols);
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
        softmax_v2_warp_per_row<<<grid, block>>>(d_input, d_output, rows, cols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    float avg_ms = milliseconds / iterations;

    std::cout << "softmax_v2_warp_per_row latency = "
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
