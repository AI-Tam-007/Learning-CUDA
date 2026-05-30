#include <cuda_runtime.h>

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

constexpr int M = 1024;
constexpr int K = 1024;
constexpr int N = 1000;

constexpr int WARMUP_TIMES = 5;
constexpr int BENCHMARK_TIMES = 10;
constexpr float ABS_EPSILON = 1e-2f;
constexpr float REL_EPSILON = 1e-3f;

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

void matmul_cpu_reference(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k) {

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;

            for (int i = 0; i < k; ++i) {
                sum += A[row * k + i] * B[i * n + col];
            }

            C[row * n + col] = sum;
        }
    }
}

void check_result(const float* cpu, const float* gpu, int count) {
    for (int i = 0; i < count; ++i) {
        const float diff = std::fabs(cpu[i] - gpu[i]);
        const float rel_diff = diff / (std::fabs(cpu[i]) + 1e-6f);

        if (diff > ABS_EPSILON && rel_diff > REL_EPSILON) {
            std::cout << "Correctness check: FAILED" << std::endl;
            std::cout << "index = " << i
                      << ", cpu = " << cpu[i]
                      << ", gpu = " << gpu[i]
                      << ", diff = " << diff
                      << ", rel_diff = " << rel_diff << std::endl;
            return;
        }
    }

    std::cout << "Correctness check: PASSED" << std::endl;
}

constexpr int BLOCK_SIZE = 32;

__global__ void matmul_v00_gpu_naive(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = sum;
}

int main() {
    constexpr int a_count = M * K;
    constexpr int b_count = K * N;
    constexpr int c_count = M * N;

    constexpr size_t a_bytes = a_count * sizeof(float);
    constexpr size_t b_bytes = b_count * sizeof(float);
    constexpr size_t c_bytes = c_count * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(a_bytes));
    float* h_B = static_cast<float*>(std::malloc(b_bytes));
    float* h_C_cpu = static_cast<float*>(std::malloc(c_bytes));
    float* h_C_gpu = static_cast<float*>(std::malloc(c_bytes));

    if (h_A == nullptr || h_B == nullptr || h_C_cpu == nullptr || h_C_gpu == nullptr) {
        std::cerr << "Host malloc failed" << std::endl;
        return EXIT_FAILURE;
    }

    load_float_bin("data/normal_1024x1024_fp32.bin", h_A, a_count);
    load_float_bin("data/normal_1024x1000_fp32.bin", h_B, b_count);

    matmul_cpu_reference(h_A, h_B, h_C_cpu, M, N, K);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A), a_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B), b_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, c_bytes));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < WARMUP_TIMES; ++i) {
        matmul_v00_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        matmul_v00_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    const float avg_ms = milliseconds / BENCHMARK_TIMES;
    const float gflops = 2.0f * M * N * K / (avg_ms * 1e6f);

    std::cout << "Kernel: matmul_v00_gpu_naive" << std::endl;
    std::cout << "A: " << M << " x " << K << std::endl;
    std::cout << "B: " << K << " x " << N << std::endl;
    std::cout << "C: " << M << " x " << N << std::endl;
    std::cout << "Average time: " << avg_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, c_bytes, cudaMemcpyDeviceToHost));
    check_result(h_C_cpu, h_C_gpu, c_count);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C_cpu);
    std::free(h_C_gpu);

    return 0;
}
