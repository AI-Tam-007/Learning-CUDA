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
    float* A,
    float* B,
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

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

#define FETCH_FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__device__ void load_tile_float4(
    float* A,
    float* B,
    float* As,
    float* Bs,
    int m,
    int n,
    int k,
    int tile,
    int block_row,
    int block_col,
    int tid,
    int thread_num) {

    for (int idx = tid; idx < BlockM * BlockK / 4; idx += thread_num) {
        const int row = idx / (BlockK / 4);
        const int col = (idx % (BlockK / 4)) * 4;
        const int global_row = block_row * BlockM + row;
        const int global_col = tile + col;

        if (global_row < m && global_col + 3 < k) {
            FETCH_FLOAT4(As[row * BlockK + col]) = FETCH_FLOAT4(A[global_row * k + global_col]);
        } else {
            for (int v = 0; v < 4; ++v) {
                if (global_row < m && global_col + v < k) {
                    As[row * BlockK + col + v] = A[global_row * k + global_col + v];
                } else {
                    As[row * BlockK + col + v] = 0.0f;
                }
            }
        }
    }

    for (int idx = tid; idx < BlockK * BlockN / 4; idx += thread_num) {
        const int row = idx / (BlockN / 4);
        const int col = (idx % (BlockN / 4)) * 4;
        const int global_row = tile + row;
        const int global_col = block_col * BlockN + col;

        if (global_row < k && global_col + 3 < n) {
            FETCH_FLOAT4(Bs[row * BlockN + col]) = FETCH_FLOAT4(B[global_row * n + global_col]);
        } else {
            for (int v = 0; v < 4; ++v) {
                if (global_row < k && global_col + v < n) {
                    Bs[row * BlockN + col + v] = B[global_row * n + global_col + v];
                } else {
                    Bs[row * BlockN + col + v] = 0.0f;
                }
            }
        }
    }
}

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__global__ void matmul_v04_double_buffering(
    float* A,
    float* B,
    float* C,
    int m,
    int n,
    int k) {

    __shared__ float As[2][BlockM * BlockK];
    __shared__ float Bs[2][BlockK * BlockN];

    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    const int tid = thread_row * blockDim.x + thread_col;
    const int thread_num = blockDim.x * blockDim.y;

    const int local_row = thread_row * ThreadM;
    const int local_col = thread_col * ThreadN;
    const int global_row_base = blockIdx.y * BlockM + local_row;
    const int global_col_base = blockIdx.x * BlockN + local_col;

    float accum[ThreadM][ThreadN] = {0.0f};

    load_tile_float4<BlockM, BlockN, BlockK, ThreadM, ThreadN>(
        A,
        B,
        As[0],
        Bs[0],
        m,
        n,
        k,
        0,
        blockIdx.y,
        blockIdx.x,
        tid,
        thread_num);
    __syncthreads();

    int read_index = 0;
    int write_index = 1;

    for (int tile = 0; tile < k; tile += BlockK) {
        const int next_tile = tile + BlockK;

        if (next_tile < k) {
            load_tile_float4<BlockM, BlockN, BlockK, ThreadM, ThreadN>(
                A,
                B,
                As[write_index],
                Bs[write_index],
                m,
                n,
                k,
                next_tile,
                blockIdx.y,
                blockIdx.x,
                tid,
                thread_num);
        }

        __syncthreads();

        for (int i = 0; i < BlockK; ++i) {
            float a_frag[ThreadM];
            float b_frag[ThreadN];

            for (int tm = 0; tm < ThreadM; ++tm) {
                a_frag[tm] = As[read_index][(local_row + tm) * BlockK + i];
            }

            for (int tn = 0; tn < ThreadN; ++tn) {
                b_frag[tn] = Bs[read_index][i * BlockN + local_col + tn];
            }

            for (int tm = 0; tm < ThreadM; ++tm) {
                for (int tn = 0; tn < ThreadN; ++tn) {
                    accum[tm][tn] += a_frag[tm] * b_frag[tn];
                }
            }
        }

        __syncthreads();
        read_index ^= 1;
        write_index ^= 1;
    }

    for (int tm = 0; tm < ThreadM; ++tm) {
        const int row = global_row_base + tm;
        if (row < m) {
            for (int tn = 0; tn < ThreadN; tn += 4) {
                const int col = global_col_base + tn;

                if (col + 3 < n) {
                    float4 out;
                    out.x = accum[tm][tn];
                    out.y = accum[tm][tn + 1];
                    out.z = accum[tm][tn + 2];
                    out.w = accum[tm][tn + 3];
                    FETCH_FLOAT4(C[row * n + col]) = out;
                } else {
                    for (int v = 0; v < 4; ++v) {
                        if (col + v < n) {
                            C[row * n + col + v] = accum[tm][tn + v];
                        }
                    }
                }
            }
        }
    }
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

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    for (int i = 0; i < WARMUP_TIMES; ++i) {
        matmul_v04_double_buffering<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        matmul_v04_double_buffering<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    const float avg_ms = milliseconds / BENCHMARK_TIMES;
    const float gflops = 2.0f * M * N * K / (avg_ms * 1e6f);

    std::cout << "Kernel: matmul_v04_double_buffering" << std::endl;
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
