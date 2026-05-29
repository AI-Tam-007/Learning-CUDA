#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

constexpr int ROWS = 1024;
constexpr int COLS = 1024;
constexpr int REPEAT_TIMES = 10;
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

void softmax_v01_cpu_reference(const float* input, float* output, int rows, int cols) {
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

void check_result(const float* output, const float* ground_truth, int count) {
    for (int i = 0; i < count; ++i) {
        const float diff = std::fabs(output[i] - ground_truth[i]);

        if (diff > EPSILON) {
            std::cout << "Correctness check: FAILED" << std::endl;
            std::cout << "index = " << i
                      << ", output = " << output[i]
                      << ", ground_truth = " << ground_truth[i]
                      << ", diff = " << diff << std::endl;
            return;
        }
    }

    std::cout << "Correctness check: PASSED" << std::endl;
}

float benchmark_v01_cpu_reference(const float* input, float* output, int rows, int cols) {
    const auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPEAT_TIMES; ++i) {
        softmax_v01_cpu_reference(input, output, rows, cols);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> elapsed_ms = end - start;

    return elapsed_ms.count() / REPEAT_TIMES;
}

int main() {
    constexpr int count = ROWS * COLS;
    constexpr size_t bytes = count * sizeof(float);

    float* input = static_cast<float*>(std::malloc(bytes));
    float* output = static_cast<float*>(std::malloc(bytes));
    float* ground_truth = static_cast<float*>(std::malloc(bytes));

    if (input == nullptr || output == nullptr || ground_truth == nullptr) {
        std::cerr << "Host malloc failed" << std::endl;
        return EXIT_FAILURE;
    }

    load_float_bin("data/normal_1024x1024_fp32.bin", input, count);
    load_float_bin("data/gt_1024x1024_fp32.bin", ground_truth, count);

    softmax_v01_cpu_reference(input, output, ROWS, COLS);
    check_result(output, ground_truth, count);

    const float avg_ms = benchmark_v01_cpu_reference(input, output, ROWS, COLS);

    std::cout << "Kernel: softmax_v01_cpu_reference" << std::endl;
    std::cout << "Rows: " << ROWS << ", Cols: " << COLS << std::endl;
    std::cout << "Average time: " << avg_ms << " ms" << std::endl;

    std::free(input);
    std::free(output);
    std::free(ground_truth);

    return 0;
}
