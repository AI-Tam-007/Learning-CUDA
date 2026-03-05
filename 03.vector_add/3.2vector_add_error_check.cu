/*
====== 向量加法(基础版) + 错误检测 ======
总结：
加入错误检测显得很专业，就和C++在创建头文件之前用： #pragma once  一样。高大上一些。

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






__global__ void vector_add(float* gpu_x, float* gpu_y, float* gpu_z, int N)    // 核函数。
{


    // =========================一维=========================
    //int glabal_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // ======================================================


    // ======================================二维======================================
    int global_idx = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;    // 计算所有线程的唯一索引。
    // ================================================================================

    if(global_idx < N)   // global_idx < N是因为main函数中设置的线程数量大于数据数量，防止数组越界。
    {
        gpu_z[global_idx] = gpu_x[global_idx] + gpu_y[global_idx];    // gpu_z = gpu_x + gpu_y
    }

}



int main()
{
    int N = 10000;  // 总共有10000个数
    int blockSize = 256;   // 假设1个block存在256个线程。
    dim3 block(blockSize);  // 启动核函数可直接用blockSize，但是用上dim3比较规范。


    // =======================================一维=======================================
    // 定义一维grid
    // int gridSize_one = (N + blockSize - 1.) / blockSize;   // 此处的1.很细节，避免整数除断。
    // 解析：(N + blockSize - 1.) / blockSize，其实就是为了定义足够的block，让线程很好地适配数据的数量。
    // 解析：(10000 + 256 - 1.)/256 ≈ 40.05859 向上取整为41，因此需要41个256大小地block。
    // 解析：41*256 = 10496个线程，但是我们只有10000个数据，因此有496个线程是闲置的。

    //dim3 grid_one(gridSize_one);   // 定义一个1维网格，其实也就是定义了gridSize_one个block，再用gridSize_one * blockSize就得到总线程数。
    // ==================================================================================



    // =======================================二维=======================================
    // 定义二维grid
    int gridSize_two = ceil(sqrt((N + blockSize - 1.) / blockSize));     // 此处的1.很细节，避免整数除断。
    //解析：sqrt((N + blockSize - 1.) / blockSize)中多了一个sqrt是因为我们定义的是一个n×n的网格，所以我们要开根号，获得x和y的数量，如上sqrt(40.05859)并向上取整为7。
    // 因此我们定义了一个7×7大小的网格grid，展平成一维就是49个block，总共存在49*256 = 12544个thread，闲置2544个线程，这比一维grid浪费的资源更多。

    
    dim3 grid_two(gridSize_two, gridSize_two);
    // ==================================================================================


    // 定义cpu_x、cpu_y用于cpu端初始化两个一维张量，cpu_z用于存储cpu端的张量和。
    float *cpu_x, *cpu_y, *cpu_z;  

    // 定义gpu_x、gpu_y用于接收cpu端传到gpu端的张量数据，gpu_z用于在gpu中存储张量和。
    float *gpu_x, *gpu_y, *gpu_z;

    // 这里为什么会又设定一个值呢？因为cpu_x对应gpu_x，cpu_y对应gpu_y，cpu_z对应cpu_x + cpu_y，gpu_z对应的gpu_x + gpu_y。
    // 这时我需要将gpu_z的结果传回cpu端，但是没有对应接收的变量了。因此我需要再定义一个变量用于接收gpu端到cpu端的结果传输。
    // 其实不定义它也可以，直接用cpu_z来接收结果，这样操作就保存不了cpu端的结果，但可以保存gpu端的输出结果。
    // 综上所述，代码总是需要重新定义一个变量用于接收cpu或gpu中的值，以便用于后续cpu结果和gpu结果的一个合格性检查。
    float *gpu_result;

    int nbytes = N * sizeof(float);  // 分配空间所需的总字节数 = 总数据量*数据类型的字节数，因为定义的数据是用的float，因此：N * sizeof(float)。

    // cpu中用C语言中的malloc分配空间。
    cpu_x = (float *)malloc(nbytes);
    cpu_y = (float *)malloc(nbytes);
    cpu_z = (float *)malloc(nbytes);

    gpu_result = (float *)malloc(nbytes);   // 它是cpu端接收gpu结果的变量，所以用cpu的C语言中的malloc分配空间，不要搞混淆了。


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
    //vector_add<<<grid_one, block>>>(gpu_x, gpu_y, gpu_z, N);
    // =======================================================


    // ============= 二维启用 ================================
    // 启动核函数(二维)。
    vector_add<<<grid_two, block >>>(gpu_x, gpu_y, gpu_z, N);
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
