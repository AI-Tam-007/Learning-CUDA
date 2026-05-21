#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>

#include <fstream>
#include <iostream>
#include <cstdlib>



#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        int row_start = row * cols;

        float max_val = -FLT_MAX;

        for (int col = 0; col < cols; col++) {
            float x = input[row_start + col];
            max_val = std::fmax(max_val, x);
        }

        float sum_val = 0.0f;

        for (int col = 0; col < cols; col++) {
            float x = input[row_start + col];
            sum_val += std::exp(x - max_val);
        }

        for (int col = 0; col < cols; col++) {
            float x = input[row_start + col];
            output[row_start + col] = std::exp(x - max_val) / sum_val;
        }
    }
}

__global__ void softmax_gpu_one_thread_per_row(const float* input, float* output, int rows, int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    int row_start = row * cols;

    // 1. Find max value of this row
    float max_val = -FLT_MAX;

    for (int col = 0; col < cols; col++) {
        float x = input[row_start + col];
        max_val = fmaxf(max_val, x);
    }

    // 2. Compute sum(exp(x - max))
    float sum_val = 0.0f;

    for (int col = 0; col < cols; col++) {
        float x = input[row_start + col];
        sum_val += expf(x - max_val);
    }

    // 3. Normalize
    for (int col = 0; col < cols; col++) {
        float x = input[row_start + col];
        output[row_start + col] = expf(x - max_val) / sum_val;
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


void check_result(const float* cpu, const float* gpu, int n) {
    const float eps = 1e-5f;

    for (int i = 0; i < n; i++) {
        float diff = std::fabs(cpu[i] - gpu[i]);

        if (diff > eps) {
            std::cout << "wrong!" << std::endl;
            std::cout << "index = " << i << std::endl;
            std::cout << "cpu = " << cpu[i] << std::endl;
            std::cout << "gpu = " << gpu[i] << std::endl;
            std::cout << "diff = " << diff << std::endl;
            return;
        }
    }

    std::cout << "right!" << std::endl;
}

int main() {
    const int rows = 128;
    const int cols = 256;
    const int N = rows * cols;

    const size_t bytes = N * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    float* h_output_cpu = static_cast<float*>(std::malloc(bytes));
    float* h_output_gpu = static_cast<float*>(std::malloc(bytes));

    if (h_input == nullptr || h_output_cpu == nullptr || h_output_gpu == nullptr) {
        std::cerr << "Host malloc failed" << std::endl;
        return EXIT_FAILURE;
    }


    load_float_bin("data/normal_128x256_fp32.bin", h_input, N);

    softmax_cpu(h_input, h_output_cpu, rows, cols);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (rows + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

    // Warmup
    for (int i = 0; i < 5; i++) {
        softmax_gpu_one_thread_per_row<<<grid, block>>>(
            d_input,
            d_output,
            rows,
            cols
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 10;

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        softmax_gpu_one_thread_per_row<<<grid, block>>>(
            d_input,
            d_output,
            rows,
            cols
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_ms = milliseconds / iterations;

    std::cout << "softmax_gpu_one_thread_per_row latency = "
              << avg_ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    check_result(h_output_cpu, h_output_gpu, N);

    std::cout << "cpu[0] = " << h_output_cpu[0] << std::endl;
    std::cout << "gpu[0] = " << h_output_gpu[0] << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    std::free(h_input);
    std::free(h_output_cpu);
    std::free(h_output_gpu);

    return 0;
}