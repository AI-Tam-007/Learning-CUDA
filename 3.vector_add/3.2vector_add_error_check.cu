/*
====== 原向量加法 + 错误检测 ======
总结：
让自己显得很专业，就和C++在创建头文件之前用： #pragma once  一样。高大上一些。


*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>   




// ===================== 错误检测 ==================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
// =================================================================






__global__ void vector_add(float* gpu_x, float* gpu_y, float* gpu_z, int N)    // 核函数，传入3个指针数组。
{
    // ============= 二维启用 ================================
    int glabal_idx = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;  // 计算每个线程的唯一索引(二维)。
    // =====================================================


    // ============= 一维启用 =================================
    //int glabal_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // ======================================================


    if(glabal_idx < N)  
    {
        gpu_z[glabal_idx] = gpu_x[glabal_idx] + gpu_y[glabal_idx];   
    }

}


int main()
{
    int N = 10000;  
    int block_num = 256;  
    


    // =============一维启用================================
    // 定义一维grid
    //int grid_num_one = ceil((N + block_num - 1.) / block_num);   // 此处的1.很细节，避免整数除断。
    // 解析：(N + block_num - 1.) / block_num，其实就是为了定义足够的block，让线程很好地适配数据的数量。
    // 解析：(10000 + 256 - 1.)/256 ≈ 40.05859 向上取整为41，因此需要41个256大小地block。
    // 解析：41*256 = 10496个线程，但是我们只有10000个数据，因此有496个线程是闲置的。

    //dim3 grid_one(grid_num_one);   // 定义一个1维网格，其实也就是定义了grid_num_one个block，再用grid_num_one * block_num就得到总线程数。
    // =====================================================




    int grid_num_two = ceil(sqrt((N + block_num - 1.) / block_num)); 


    
    dim3 grid_two(grid_num_two, grid_num_two);



    float *cpu_x, *cpu_y, *cpu_z;  

    float *gpu_x, *gpu_y, *gpu_z;


    float *gpu_result;

    int nbytes = N * sizeof(float);  

    cpu_x = (float *)malloc(nbytes);
    cpu_y = (float *)malloc(nbytes);
    cpu_z = (float *)malloc(nbytes);

    gpu_result = (float *)malloc(nbytes);   

// ===================== 错误检测 ==================================
    gpuErrchk(cudaMalloc((void**)&gpu_x, nbytes));
    gpuErrchk(cudaMalloc((void**)&gpu_y, nbytes));
    gpuErrchk(cudaMalloc((void**)&gpu_z, nbytes));
// ================================================================



    // 在cpu中对数据进行初始化。
    for(int i = 0; i < N; i ++)
    {
        cpu_x[i] = 1;
        cpu_y[i] = 1;
        cpu_z[i] = cpu_x[i] + cpu_y[i];
    }


    // ===================== 错误检测 ==================================
    gpuErrchk(cudaMemcpy(gpu_x, cpu_x, nbytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_y, cpu_y, nbytes, cudaMemcpyHostToDevice));
    // ================================================================




    float milliseconds = 0.0; // 它必须设定为float类型,因为后面的cudaEventElapsedTime算出的时间是毫秒级别。

    cudaEvent_t start, stop; 


    cudaEventCreate(&start); 
    cudaEventCreate(&stop);


    cudaEventRecord(start);

     // ============= 一维启用 ================================
    // 启动核函数(一维)。
    //vector_add<<<grid_one, block_num>>>(gpu_x, gpu_y, gpu_z, N);
    // =======================================================


    // ============= 二维启用 ================================
    // 启动核函数(二维)。
    vector_add<<<grid_two, block_num >>>(gpu_x, gpu_y, gpu_z, N);
    // ======================================================


    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    // ===================== 错误检测 ==================================
    gpuErrchk(cudaMemcpy(gpu_result, gpu_z, nbytes, cudaMemcpyDeviceToHost));
    // ================================================================

    int error_count = 0; 

    for(int i = 0; i < N; i++)
    {
        if(fabs(cpu_z[i] - gpu_result[i]) > 1e-6)
        {
            
            error_count += 1;
        }
    }

    if(error_count == 0)
    {
        printf("Result right\n");
    }
    

    free(cpu_x);
    free(cpu_y);
    free(cpu_z);
    free(gpu_result);

    // ===================== 错误检测 ==================================
    gpuErrchk(cudaFree(gpu_x));
    gpuErrchk(cudaFree(gpu_y));
    gpuErrchk(cudaFree(gpu_z));
    // ================================================================

    return 0;



}
