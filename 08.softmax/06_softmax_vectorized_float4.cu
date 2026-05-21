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
// Error Checking
// ================================================================

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
cudaError_t err = call;                                                    \
if (err != cudaSuccess) {                                                  \
std::cout << "CUDA error: " << cudaGetErrorString(err)                 \
<< " at line " << __LINE__ << std::endl;                    \
return -1;                                                             \
}                                                                          \
} while (0)

// ================================================================
// Utility Types
// ================================================================

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

// ================================================================
// CPU Reference
// ================================================================

void softmax_cpu(float* input, float* output, int rows, int cols) {

    for (int row = 0; row < rows; row++) {

        float max_val = -FLT_MAX;

        for (int col = 0; col < cols; col++) {
            max_val = fmaxf(max_val, input[row * cols + col]);
        }

        float sum_val = 0.0f;

        for (int col = 0; col < cols; col++) {
            sum_val += expf(input[row * cols + col] - max_val);
        }

        for (int col = 0; col < cols; col++) {
            output[row * cols + col] =
            expf(input[row * cols + col] - max_val) / sum_val;
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
// 每个线程处理 Cols_Per_Thread 个元素
// 每次向量化读写 Pack_Size 个 float

// ================================================================
// CUDA Kernel
// ================================================================

template<int Pack_Size, int Cols_Per_Thread>
__global__ void softmax_v4_warp_per_row_vectorized(
    const float* d_input,
    float* d_output,
    int rows, 
    int cols
) {
    static_assert(Cols_Per_Thread % Pack_Size == 0, 
        "Cols_Per_Thread must be divisible by Pack_Size");

    using VecType = VectorType<float, Pack_Size>;

    constexpr int Num_Packs = Cols_Per_Thread / Pack_Size;

    int lane = threadIdx.x; // 0 ~ 31
    int warp_in_block = threadIdx.y; // block 内第几个 warp

    int row = blockIdx.x * blockDim.y + warp_in_block;

    if (row >= rows) {
        return;
    }

    int row_start = row * cols;

    float buf[Cols_Per_Thread];

    float local_max = -FLT_MAX;

    // step 1:
    // 从 global memory 向量化读取到寄存器 buf，同时求 local_max
#pragma unroll
    for (int pack_id = 0; pack_id < Num_Packs; pack_id++) {

        int base_col = (pack_id * WARP_SIZE + lane) * Pack_Size;
        int buf_offset = pack_id * Pack_Size;

        if (base_col + Pack_Size <= cols) {

            const VecType* src_vec =
            reinterpret_cast<const VecType*>(
                d_input + row_start + base_col
            );

            VecType tmp = *src_vec;

#pragma unroll
            for (int i = 0; i < Pack_Size; i++) {
                float x = tmp.val[i];
                buf[buf_offset + i] = x;
                local_max = fmaxf(local_max, x);
            }

        } else {

#pragma unroll
        for (int i = 0; i < Pack_Size; i++) {

            int col = base_col + i;

            if (col < cols) {
                float x = d_input[row_start + col];
                buf[buf_offset + i] = x;
                local_max = fmaxf(local_max, x);
            } else {
            buf[buf_offset + i] = -FLT_MAX;
        }
    }
}
}

// step 2:
float max_val = warp_reduce_max(local_max);

// __shfl_down_sync reduce 后，完整结果只保证在 lane 0
// 所以从 lane 0 广播给整个 warp
max_val = __shfl_sync(0xffffffff, max_val, 0);

// step 3:
// 使用寄存器 buf 计算 expf(x - max)，并求 local_sum
float local_sum = 0.0f;

#pragma unroll
for (int i = 0; i < Cols_Per_Thread; i++) {
    float e = expf(buf[i] - max_val);
    buf[i] = e;
    local_sum += e;
}

// step 4:
float sum_val = warp_reduce_sum(local_sum);

// 同样从 lane 0 广播给整个 warp
sum_val = __shfl_sync(0xffffffff, sum_val, 0);

// step 5:
// softmax 结果写回，优先使用向量化 store
#pragma unroll
for (int pack_id = 0; pack_id < Num_Packs; pack_id++) {

    int base_col = (pack_id * WARP_SIZE + lane) * Pack_Size;
    int buf_offset = pack_id * Pack_Size;

    VecType tmp;

#pragma unroll
    for (int i = 0; i < Pack_Size; i++) {
        tmp.val[i] = buf[buf_offset + i] / sum_val;
    }

    if (base_col + Pack_Size <= cols) {

        VecType* dst_vec =
        reinterpret_cast<VecType*>(
            d_output + row_start + base_col
        );

        *dst_vec = tmp;

    } else {

#pragma unroll
    for (int i = 0; i < Pack_Size; i++) {

        int col = base_col + i;

        if (col < cols) {
            d_output[row_start + col] = tmp.val[i];
        }
    }
}
}
}

// ================================================================
// Main
// ================================================================

int main() {

    int rows = 8192;
    int cols = 4096;
    int N = rows * cols;

    float* input = (float*)malloc(N * sizeof(float));
    float* output_cpu = (float*)malloc(N * sizeof(float));
    float* output_gpu2cpu = (float*)malloc(N * sizeof(float));

    if (!input || !output_cpu || !output_gpu2cpu) {
        std::cout << "host malloc failed!" << std::endl;
        return -1;
    }

    load_float_bin("data/normal_8192x4096_fp32.bin", input, N);

    softmax_cpu(input, output_cpu, rows, cols);

    float* d_input;
    float* d_output;

    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_input, 
        input, 
        N * sizeof(float), 
        cudaMemcpyHostToDevice
    ));
    dim3 block(32, 8);
    dim3 grid((rows + block.y - 1) / block.y);

    // 当前 cols = 1024
    // 每个线程处理 1024 / 32 = 32 个 float
    // 每次向量化读写 4 个 float
    constexpr int Pack_Size = 4;
    constexpr int Cols_Per_Thread = 128;

    // warmup
    softmax_v4_warp_per_row_vectorized<Pack_Size, Cols_Per_Thread>
    <<<grid, block>>>(d_input, d_output, rows, cols);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iterations = 10;
    float milliseconds = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        softmax_v4_warp_per_row_vectorized<Pack_Size, Cols_Per_Thread>
        <<<grid, block>>>(d_input, d_output, rows, cols);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float avg_ms = milliseconds / iterations;

    std::cout << "softmax_v4_warp_per_row_vectorized latency = "
    << avg_ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(
        output_gpu2cpu, 
        d_output, 
        N * sizeof(float), 
        cudaMemcpyDeviceToHost
    ));

    check_cpu(output_cpu, output_gpu2cpu, N);

    std::cout << "cpu[0] = " << output_cpu[0] << std::endl;
    std::cout << "gpu[0] = " << output_gpu2cpu[0] << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    free(input);
    free(output_cpu);
    free(output_gpu2cpu);

    return 0;
}
