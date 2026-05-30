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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FETCH_FLOAT4_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])

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

void matmul_cpu_reference(const float* A, const float* B, float* C, int m, int n, int k) {
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

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__device__ __forceinline__ void load_first_tile(
    const float* A,
    const float* B,
    float* As,
    float* Bs,
    int m,
    int n,
    int k,
    int block_row,
    int block_col,
    int tid) {

    constexpr int thread_num = (BlockM / ThreadM) * (BlockN / ThreadN);
    constexpr int ldg_a_num = BlockK * BlockM / thread_num / 4;
    constexpr int ldg_b_num = BlockK * BlockN / thread_num / 4;

    const int a_tile_row = tid / (BlockK / 4);
    const int a_tile_col = (tid % (BlockK / 4)) * 4;
    constexpr int a_tile_stride = BlockM / ldg_a_num;

    const int b_tile_row = tid / (BlockN / 4);
    const int b_tile_col = (tid % (BlockN / 4)) * 4;
    constexpr int b_tile_stride = BlockK / ldg_b_num;

#pragma unroll
    for (int i = 0; i < BlockM; i += a_tile_stride) {
        const int global_row = block_row * BlockM + a_tile_row + i;
        const int global_col = a_tile_col;

        if (global_row < m && global_col + 3 < k) {
            float4 value = FETCH_FLOAT4_CONST(A[global_row * k + global_col]);
            As[OFFSET(a_tile_col,     a_tile_row + i, BlockM)] = value.x;
            As[OFFSET(a_tile_col + 1, a_tile_row + i, BlockM)] = value.y;
            As[OFFSET(a_tile_col + 2, a_tile_row + i, BlockM)] = value.z;
            As[OFFSET(a_tile_col + 3, a_tile_row + i, BlockM)] = value.w;
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int col = global_col + v;
                As[OFFSET(a_tile_col + v, a_tile_row + i, BlockM)] =
                    (global_row < m && col < k) ? A[global_row * k + col] : 0.0f;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < BlockK; i += b_tile_stride) {
        const int global_row = b_tile_row + i;
        const int global_col = block_col * BlockN + b_tile_col;

        if (global_row < k && global_col + 3 < n) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BlockN)]) =
                FETCH_FLOAT4_CONST(B[global_row * n + global_col]);
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int col = global_col + v;
                Bs[OFFSET(b_tile_row + i, b_tile_col + v, BlockN)] =
                    (global_row < k && col < n) ? B[global_row * n + col] : 0.0f;
            }
        }
    }
}

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__device__ __forceinline__ void prefetch_next_tile_to_registers(
    const float* A,
    const float* B,
    float* ldg_a_reg,
    float* ldg_b_reg,
    int m,
    int n,
    int k,
    int tile,
    int block_row,
    int block_col,
    int tid) {

    constexpr int thread_num = (BlockM / ThreadM) * (BlockN / ThreadN);
    constexpr int ldg_a_num = BlockK * BlockM / thread_num / 4;
    constexpr int ldg_b_num = BlockK * BlockN / thread_num / 4;

    const int a_tile_row = tid / (BlockK / 4);
    const int a_tile_col = (tid % (BlockK / 4)) * 4;
    constexpr int a_tile_stride = BlockM / ldg_a_num;

    const int b_tile_row = tid / (BlockN / 4);
    const int b_tile_col = (tid % (BlockN / 4)) * 4;
    constexpr int b_tile_stride = BlockK / ldg_b_num;

#pragma unroll
    for (int i = 0; i < BlockM; i += a_tile_stride) {
        const int ldg_index = i / a_tile_stride * 4;
        const int global_row = block_row * BlockM + a_tile_row + i;
        const int global_col = tile + a_tile_col;

        if (global_row < m && global_col + 3 < k) {
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4_CONST(A[global_row * k + global_col]);
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int col = global_col + v;
                ldg_a_reg[ldg_index + v] =
                    (global_row < m && col < k) ? A[global_row * k + col] : 0.0f;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < BlockK; i += b_tile_stride) {
        const int ldg_index = i / b_tile_stride * 4;
        const int global_row = tile + b_tile_row + i;
        const int global_col = block_col * BlockN + b_tile_col;

        if (global_row < k && global_col + 3 < n) {
            FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                FETCH_FLOAT4_CONST(B[global_row * n + global_col]);
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int col = global_col + v;
                ldg_b_reg[ldg_index + v] =
                    (global_row < k && col < n) ? B[global_row * n + col] : 0.0f;
            }
        }
    }
}

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__device__ __forceinline__ void write_registers_to_shared(
    const float* ldg_a_reg,
    const float* ldg_b_reg,
    float* As,
    float* Bs,
    int tid) {

    constexpr int thread_num = (BlockM / ThreadM) * (BlockN / ThreadN);
    constexpr int ldg_a_num = BlockK * BlockM / thread_num / 4;
    constexpr int ldg_b_num = BlockK * BlockN / thread_num / 4;

    const int a_tile_row = tid / (BlockK / 4);
    const int a_tile_col = (tid % (BlockK / 4)) * 4;
    constexpr int a_tile_stride = BlockM / ldg_a_num;

    const int b_tile_row = tid / (BlockN / 4);
    const int b_tile_col = (tid % (BlockN / 4)) * 4;
    constexpr int b_tile_stride = BlockK / ldg_b_num;

#pragma unroll
    for (int i = 0; i < BlockM; i += a_tile_stride) {
        const int ldg_index = i / a_tile_stride * 4;

        As[OFFSET(a_tile_col,     a_tile_row + i, BlockM)] = ldg_a_reg[ldg_index];
        As[OFFSET(a_tile_col + 1, a_tile_row + i, BlockM)] = ldg_a_reg[ldg_index + 1];
        As[OFFSET(a_tile_col + 2, a_tile_row + i, BlockM)] = ldg_a_reg[ldg_index + 2];
        As[OFFSET(a_tile_col + 3, a_tile_row + i, BlockM)] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BlockK; i += b_tile_stride) {
        const int ldg_index = i / b_tile_stride * 4;

        FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BlockN)]) =
            FETCH_FLOAT4_CONST(ldg_b_reg[ldg_index]);
    }
}

template<int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__global__ void __launch_bounds__(256) matmul_v04_double_buffering(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k) {

    __shared__ float As[2][BlockK * BlockM];
    __shared__ float Bs[2][BlockK * BlockN];

    const int tid = threadIdx.x;

    constexpr int block_row_thread = BlockN / ThreadN;
    const int local_col = (tid % block_row_thread) * ThreadN;
    const int local_row = (tid / block_row_thread) * ThreadM;

    const int global_row_base = blockIdx.y * BlockM + local_row;
    const int global_col_base = blockIdx.x * BlockN + local_col;

    constexpr int thread_num = (BlockM / ThreadM) * (BlockN / ThreadN);
    constexpr int ldg_a_num = BlockK * BlockM / thread_num / 4;
    constexpr int ldg_b_num = BlockK * BlockN / thread_num / 4;

    float accum[ThreadM][ThreadN] = {0.0f};
    float ldg_a_reg[4 * ldg_a_num] = {0.0f};
    float ldg_b_reg[4 * ldg_b_num] = {0.0f};
    float a_frag[2][ThreadM] = {0.0f};
    float b_frag[2][ThreadN] = {0.0f};

    load_first_tile<BlockM, BlockN, BlockK, ThreadM, ThreadN>(
        A, B, As[0], Bs[0], m, n, k, blockIdx.y, blockIdx.x, tid);
    __syncthreads();

#pragma unroll
    for (int tm = 0; tm < ThreadM; tm += 4) {
        FETCH_FLOAT4(a_frag[0][tm]) =
            FETCH_FLOAT4(As[0][OFFSET(0, local_row + tm, BlockM)]);
    }

#pragma unroll
    for (int tn = 0; tn < ThreadN; tn += 4) {
        FETCH_FLOAT4(b_frag[0][tn]) =
            FETCH_FLOAT4(Bs[0][OFFSET(0, local_col + tn, BlockN)]);
    }

    int write_index = 1;
    int tile = 0;

    do {
        tile += BlockK;

        if (tile < k) {
            prefetch_next_tile_to_registers<BlockM, BlockN, BlockK, ThreadM, ThreadN>(
                A, B, ldg_a_reg, ldg_b_reg, m, n, k, tile, blockIdx.y, blockIdx.x, tid);
        }

        const int read_index = write_index ^ 1;

#pragma unroll
        for (int bk = 0; bk < BlockK - 1; ++bk) {
#pragma unroll
            for (int tm = 0; tm < ThreadM; tm += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) & 1][tm]) =
                    FETCH_FLOAT4(As[read_index][OFFSET(bk + 1, local_row + tm, BlockM)]);
            }

#pragma unroll
            for (int tn = 0; tn < ThreadN; tn += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) & 1][tn]) =
                    FETCH_FLOAT4(Bs[read_index][OFFSET(bk + 1, local_col + tn, BlockN)]);
            }

#pragma unroll
            for (int tm = 0; tm < ThreadM; ++tm) {
#pragma unroll
                for (int tn = 0; tn < ThreadN; ++tn) {
                    accum[tm][tn] += a_frag[bk & 1][tm] * b_frag[bk & 1][tn];
                }
            }
        }

        if (tile < k) {
            write_registers_to_shared<BlockM, BlockN, BlockK, ThreadM, ThreadN>(
                ldg_a_reg, ldg_b_reg, As[write_index], Bs[write_index], tid);
            __syncthreads();

#pragma unroll
            for (int tm = 0; tm < ThreadM; tm += 4) {
                FETCH_FLOAT4(a_frag[0][tm]) =
                    FETCH_FLOAT4(As[write_index][OFFSET(0, local_row + tm, BlockM)]);
            }

#pragma unroll
            for (int tn = 0; tn < ThreadN; tn += 4) {
                FETCH_FLOAT4(b_frag[0][tn]) =
                    FETCH_FLOAT4(Bs[write_index][OFFSET(0, local_col + tn, BlockN)]);
            }

            write_index ^= 1;
        }

#pragma unroll
        for (int tm = 0; tm < ThreadM; ++tm) {
#pragma unroll
            for (int tn = 0; tn < ThreadN; ++tn) {
                accum[tm][tn] += a_frag[(BlockK - 1) & 1][tm] *
                                 b_frag[(BlockK - 1) & 1][tn];
            }
        }
    } while (tile < k);

#pragma unroll
    for (int tm = 0; tm < ThreadM; ++tm) {
        const int row = global_row_base + tm;

        if (row < m) {
#pragma unroll
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
#pragma unroll
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

    dim3 block((BM / TM) * (BN / TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < WARMUP_TIMES; ++i) {
        matmul_v04_double_buffering<BM, BN, BK, TM, TN><<<grid, block>>>(
            d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_TIMES; ++i) {
        matmul_v04_double_buffering<BM, BN, BK, TM, TN><<<grid, block>>>(
            d_A, d_B, d_C, M, N, K);
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
