#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <cstdlib>



#define NUM_ROWS 1024
#define NUM_COLS 1000
#define REPEAT_TIMES 10
#define EPSILON 1e-5f



// ================================================================
// CPU版本
// ================================================================

void softmax_cpu(float* input, float* output, int rows, int cols){

    for(int j = 0; j < rows; j++){

        float total = 0.0f;
        float max_val = -FLT_MAX;   // softmax可能存在正值负值，取最小值即可

        for(int i = 0; i < cols; i++){    // 求一行的元素最大值max
            max_val = fmaxf(input[j * cols + i], max_val);
        }

        for(int i = 0; i < cols; i++){   // 求一行的元素和sum
            total += expf(input[j * cols + i] - max_val);
        }

        for(int i = 0; i < cols; i++){  // 计算每个元素的softmax输出
            output[j * cols + i] = expf(input[j * cols + i] - max_val) / total;
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
// 检查计算结果是否正确？
// ================================================================

void check_cpu(float* output, float *groundtrue, int N){

    for(int i = 0; i < N; i++){

        if(fabsf(output[i] - groundtrue[i]) > EPSILON){
            std::cout << "Correctness check: FAILED" << std::endl;
            return;
        }
    }

    std::cout << "Correctness check: PASSED" << std::endl;
}

// ================================================================
// 计算cpu耗时
// ================================================================

float benchmark_cpu(float* input, float* output, int rows, int cols){

    auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间

    for(int i = 0; i < REPEAT_TIMES; i++){    // 循环运行10次
        softmax_cpu(input, output, rows, cols);
    }

    auto end = std::chrono::high_resolution_clock::now();   // 记录结束时间

    std::chrono::duration<float, std::milli> elapsed_ms = end - start;

    return elapsed_ms.count() / REPEAT_TIMES;  // 得到平均时间
}



int main(){

    const int rows = NUM_ROWS;
    const int cols = NUM_COLS;
    const int N = rows * cols;

    float* input = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    float* gt = (float*)malloc(N * sizeof(float));


    load_float_bin("data/normal_1024x1024_fp32.bin", input, N);
    load_float_bin("data/gt_1024x1024_fp32.bin", gt, N);

    softmax_cpu(input, output, rows, cols);

    check_cpu(output, gt, N);

    float avg_ms = benchmark_cpu(input, output, rows, cols);

    std::cout << "Kernel: softmax_cpu_reference" << std::endl;
    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;
    std::cout << "Average time: " << avg_ms << " ms" << std::endl;

    free(input);
    free(output);

    return 0;
}
