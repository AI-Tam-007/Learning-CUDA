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
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int ROWS = 8192;
constexpr int COLS = 4096;
constexpr int BLOCK_SIZE = 256;
constexpr int WARMUP_TIMES = 1;
constexpr int BENCHMARK_TIMES = 10;
constexpr float EPSILON = 1e-5f;

void load_float_bin(const char* filename, float* data, int count) {
    std::ifstream fin(filename, std::ios::binary);

    if (!fin) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fin.read(reinterpret_cast<char*>(data), count * sizeof(float));

    if (!fin) {
        std::cerr << "Failed to read enough data from file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void softmax_cpu_reference(const float* input, float* output, int rows, int cols) {
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
            output[row_start + col] = std::exp(input[row_start + col] - max_val) / sum_val;
        }
    }
}

template<int BlockSize, int NumCols>
__global__ void softmax_v06_block_shared_cache(
    const float* input,
    float* output,
    int rows) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    __shared__ float row_cache[NumCols];
    __shared__ float reduce_buf[BlockSize];

    const int row_start = row * NumCols;

    for (int col = tid; col < NumCols; col += BlockSize) {
        row_cache[col] = input[row_start + col];
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int col = tid; col < NumCols; col += BlockSize) {
        local_max = fmaxf(local_max, row_cache[col]);
    }

    reduce_buf[tid] = local_max;
    __syncthreads();

    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }

    const float max_val = reduce_buf[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < NumCols; col += BlockSize) {
        local_sum += expf(row_cache[col] - max_val);
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }

    const float sum_val = reduce_buf[0];
    __syncthreads();

    for (int col = tid; col < NumCols; col += BlockSize) {
        output[row_start + col] = expf(row_cache[col] - max_val) / sum_val;
    }
}

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

    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));

    if (BLOCK_SIZE > device_prop.maxThreadsPerBlock) {
        std::cerr << "BLOCK_SIZE exceeds maxThreadsPerBlock" << std::endl;
        return EXIT_FAILURE;
    }

    const size_t required_shared_memory = (COLS + BLOCK_SIZE) * sizeof(float);
    if (required_shared_memory > static_cast<size_t>(device_prop.sharedMemPerBlock)) {
        std::cerr << "Required shared memory exceeds sharedMemPerBlock" << std::endl;
        return EXIT_FAILURE;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(ROWS);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < WARMUP_TIMES; ++i) {
        softmax_v06_block_shared_cache<BLOCK_SIZE, COLS><<<grid, block>>>(
            d_input,
            d_output,
            ROWS);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        softmax_v06_block_shared_cache<BLOCK_SIZE, COLS><<<grid, block>>>(
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

    std::cout << "Kernel: softmax_v06_block_shared_cache" << std::endl;
    std::cout << "Rows: " << ROWS << ", Cols: " << COLS << std::endl;
    std::cout << "Average time: " << avg_ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    check_result(h_output_cpu, h_output_gpu, count);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    std::free(h_input);
    std::free(h_output_cpu);
    std::free(h_output_gpu);

    return 0;
}
