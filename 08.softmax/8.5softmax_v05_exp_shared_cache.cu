#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int ROWS = 8192;
constexpr int COLS = 4096;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;


constexpr int WARMUP_TIMES = 1;
constexpr int BENCHMARK_TIMES = 10;

constexpr float EPSILON = 1e-5f;


// ================================================================
// Load binary data
// ================================================================

void load_float_bin(const char* filename, float* data, int count) {
    std::ifstream fin(filename, std::ios::binary);

    if (!fin) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fin.read(reinterpret_cast<char*>(data), count * sizeof(float));

    if (!fin) {
        std::cerr << "Failed to read enough data from file: "
                  << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


// ================================================================
// CPU reference
// ================================================================

void softmax_cpu_reference(
    const float* input,
    float* output,
    int rows,
    int cols) {

    for (int row = 0; row < rows; ++row) {
        const int row_start = row * cols;

        float max_val = -FLT_MAX;

        for (int col = 0; col < cols; ++col) {
            max_val = std::fmax(max_val, input[row_start + col]);
        }

        float sum_val = 0.0f;

        for (int col = 0; col < cols; ++col) {
            sum_val += std::exp(input[row_start + col] - max_val);
        }

        for (int col = 0; col < cols; ++col) {
            output[row_start + col] =
                std::exp(input[row_start + col] - max_val) / sum_val;
        }
    }
}


// ================================================================
// Warp reduce
// ================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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
// 05: exp(x - max) shared memory cache
//
// block = (32, 8)
// 一个 block 处理一行
//
// 优化点：
// 1. 先求每一行的 max
// 2. 计算 exp(x - max)
// 3. 把 exp(x - max) 存入 shared memory
// 4. 求 sum(exp(x - max))
// 5. 写回时直接复用 shared memory 中的 exp 结果
//
// 避免：
// output 阶段再次计算 expf(x - max)
// ================================================================

template<int WarpsPerBlock, int Cols>
__global__ void softmax_v05_exp_shared_cache(
    const float* input,
    float* output,
    int rows) {

    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * WARP_SIZE + lane_id;

    constexpr int block_size = WARP_SIZE * WarpsPerBlock;

    const int row_start = row * Cols;

    __shared__ float warp_max[WarpsPerBlock];
    __shared__ float warp_sum[WarpsPerBlock];

    // 每个 block 处理一行，所以这里缓存当前行的 exp(x - max)
    __shared__ float exp_cache[Cols];


    // ============================================================
    // 1. 求当前行 max
    // ============================================================

    float local_max = -FLT_MAX;

    for (int col = tid; col < Cols; col += block_size) {
        local_max = fmaxf(local_max, input[row_start + col]);
    }

    local_max = warp_reduce_max(local_max);

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        float block_max = warp_max[0];

        for (int i = 1; i < WarpsPerBlock; ++i) {
            block_max = fmaxf(block_max, warp_max[i]);
        }

        warp_max[0] = block_max;
    }

    __syncthreads();

    const float max_val = warp_max[0];


    // ============================================================
    // 2. 计算 exp(x - max)，写入 shared memory，同时求局部 sum
    // ============================================================

    float local_sum = 0.0f;

    for (int col = tid; col < Cols; col += block_size) {
        const float exp_val = expf(input[row_start + col] - max_val);

        exp_cache[col] = exp_val;
        local_sum += exp_val;
    }

    __syncthreads();


    // ============================================================
    // 3. block 内 reduce sum
    // ============================================================

    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        float block_sum = warp_sum[0];

        for (int i = 1; i < WarpsPerBlock; ++i) {
            block_sum += warp_sum[i];
        }

        warp_sum[0] = block_sum;
    }

    __syncthreads();

    const float sum_val = warp_sum[0];


    // ============================================================
    // 4. 写回 softmax
    //
    // 这里不再重新计算 expf(input - max)
    // 直接使用 shared memory 里缓存好的 exp_cache
    // ============================================================

    for (int col = tid; col < Cols; col += block_size) {
        output[row_start + col] = exp_cache[col] / sum_val;
    }
}


// ================================================================
// Check result
// ================================================================

void check_result(const float* cpu, const float* gpu, int count) {
    for (int i = 0; i < count; ++i) {
        const float diff = std::fabs(cpu[i] - gpu[i]);

        if (diff > EPSILON) {
            std::cout << "Correctness check: FAILED" << std::endl;
            std::cout << "index = " << i
                      << ", cpu = " << cpu[i]
                      << ", gpu = " << gpu[i]
                      << ", diff = " << diff << std::endl;
            return;
        }
    }

    std::cout << "Correctness check: PASSED" << std::endl;
}


// ================================================================
// Main
// ================================================================

int main() {
    constexpr int count = ROWS * COLS;
    constexpr size_t bytes = count * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    float* h_output_cpu = static_cast<float*>(std::malloc(bytes));
    float* h_output_gpu = static_cast<float*>(std::malloc(bytes));

    if (h_input == nullptr || h_output_cpu == nullptr || h_output_gpu == nullptr) {
        std::cerr << "Host malloc failed" << std::endl;
        return EXIT_FAILURE;
    }

    load_float_bin("data/normal_8192x4096_fp32.bin", h_input, count);

    softmax_cpu_reference(h_input, h_output_cpu, ROWS, COLS);


    // ============================================================
    // Check shared memory usage
    // ============================================================

    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));

    constexpr size_t required_shared_memory =
        COLS * sizeof(float)
        + WARPS_PER_BLOCK * sizeof(float)
        + WARPS_PER_BLOCK * sizeof(float);

    if (required_shared_memory >
        static_cast<size_t>(device_prop.sharedMemPerBlock)) {

        std::cerr << "Required shared memory exceeds sharedMemPerBlock"
                  << std::endl;

        std::cerr << "Required shared memory: "
                  << required_shared_memory << " bytes" << std::endl;

        std::cerr << "Device sharedMemPerBlock: "
                  << device_prop.sharedMemPerBlock << " bytes" << std::endl;

        return EXIT_FAILURE;
    }


    // ============================================================
    // Allocate device memory
    // ============================================================

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice));


    // ============================================================
    // Launch config
    // ============================================================

    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 grid(ROWS);


    // ============================================================
    // Warmup
    // ============================================================

    for (int i = 0; i < WARMUP_TIMES; ++i) {
        softmax_v05_exp_shared_cache<WARPS_PER_BLOCK, COLS><<<grid, block>>>(
            d_input,
            d_output,
            ROWS);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // ============================================================
    // Benchmark
    // ============================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        softmax_v05_exp_shared_cache<WARPS_PER_BLOCK, COLS><<<grid, block>>>(
            d_input,
            d_output,
            ROWS);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    const float avg_ms = milliseconds / BENCHMARK_TIMES;

    std::cout << "Kernel: softmax_v05_exp_shared_cache" << std::endl;
    std::cout << "Rows: " << ROWS << ", Cols: " << COLS << std::endl;
    std::cout << "Average time: " << avg_ms << " ms" << std::endl;


    // ============================================================
    // Copy result back and check
    // ============================================================

    CHECK_CUDA(cudaMemcpy(
        h_output_gpu,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    check_result(h_output_cpu, h_output_gpu, count);


    // ============================================================
    // Free
    // ============================================================

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    std::free(h_input);
    std::free(h_output_cpu);
    std::free(h_output_gpu);

    return 0;
}