/*
====== 规约求和(基础版) ======
reduce的定义：并不是单纯的求和，是给N个数值，对它们做累计的算术操作，例如求和、最大值、最小值、均值、异或这一类操作的统称

*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void sum(int *gpu_arr,int *gpu_sum, int N)
{
    // int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int sum = 0;    // 这个很重要，如果直接用*gpu_sum += gpu_arr[i]，多个线程同时读写全局内存时，如果没有同步机制（如原子操作），会导致竞态条件（Race Condition）。
    for(int i = 0; i < N; i++)
    {
        sum += gpu_arr[i];
    }
    *gpu_sum = sum;

}

bool CheckResult(int *cpu_sum, int *cpu_right_sum) 
{
    // int类型的情况
    return (*cpu_sum == *cpu_right_sum);

    // float类型的情况
    // return fabs(*cpu_sum - cpu_right_sum) <= (1e-6f) * fmax(1.0f, fmax(fabs(*cpu_sum), fabs(cpu_right_sum)));

    // double类型的情况
    // return fabs(*cpu_sum - cpu_right_sum) <= (1e-12) * fmax(1.0f, fmax(fabs(*cpu_sum), fabs(cpu_right_sum)));
}


int main()
{

//获取参数量
    int N = 25600000;   // 计算两千五百六十万个数累加
    
// 指定GPU，获取GPU信息
    cudaSetDevice(0);    // 指定使用的CUDA设备
    cudaDeviceProp deviceProp;  // 存储 GPU 设备的各种属性信息
    cudaGetDeviceProperties(&deviceProp, 0);   // 把GPU设备0的详细属性填充到deviceProp结构体中
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", 0, deviceProp.name, deviceProp.major, deviceProp.minor);


// 设定参数量的存储空间大小
    int nbytes = N * sizeof(int);
    int nbytes_int = 1 * sizeof(int);  // 用于定义1个int类型的数组长度，即arr[0]，数组长度为1，故只有索引0；



// 在CPU端分配存储空间
    // 在堆中分配内存
    int *cpu_arr = (int *)malloc(nbytes);   // cpu(host)端分配内存---系统内存
    int *cpu_sum = (int *)malloc(nbytes_int);  // cpu(host)端分配内存---系统内存
    int *cpu_right_sum = (int *)malloc(nbytes_int);
    
    *cpu_right_sum = 0;   // 将单个数值当作值进行初始化

    // cpu初始化并计算数组和
    for(int i = 0; i < N; i++)
    {
        cpu_arr[i] = 1;    //等价于*(cpu_arr + i)
        *cpu_right_sum += cpu_arr[i];  // *cpu_right_sum等价于cpu_right_sum[0],为了确保可读性，最好是用*cpu_right_sum代表单个值的数组
    }


// 设定grid和block大小
    /*定义GPU的grid和block*/
    int block_num = 256;   // 定义一个block大小，即blockDim.x大小，也就是thread数量
    int grid_num = (N + block_num - 1.) / block_num;   // 定义一个grid大小，即gridDim.x，也就是block数量。如果是一维，即表示grid_num个block，如果是二维，即表示grid_num*grid_num个block
    dim3 block(block_num);  // 一维，即一个block中包含block_num个线程
    dim3 grid(grid_num);   // 一维，即一个grid中包含grid_num个block
    

// 在GPU端分配存储空间
    // 在显存中分配内存
    int *gpu_arr, *gpu_sum; 
    cudaMalloc((void **)&gpu_arr, nbytes);      // gpu(device)端分配内存---系统显存
    cudaMalloc((void **)&gpu_sum, nbytes_int);  // gpu(device)端分配内存---系统显存


    

// 将CPU端的数据拷贝到GPU端
    cudaMemcpy(gpu_arr, cpu_arr, nbytes, cudaMemcpyHostToDevice);  // 将cpu(host)端的数据转到gpu(device)端

// 计算核函数运行时间
    float milliseconds = 0;   // 计算gpu运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


// 启动核函数
    sum<<<1, 1>>>(gpu_arr, gpu_sum, N);   // 当grid、block均为1时表示串行运行。



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

// 将GPU端的数据转移到CPU端
    cudaMemcpy(cpu_sum, gpu_sum, nbytes_int, cudaMemcpyDeviceToHost);


    printf("cpu_right_sum:%d \n", *cpu_right_sum);
    printf("cpu_sum:%d \n", *cpu_sum);


// 验证CPU和GPU计算出的数据是否一致
    bool is_right = CheckResult(cpu_sum, cpu_right_sum);
    if(is_right) 
    {
        printf("the ans is right\n");
    } 
    else 
    {
        printf("the ans is wrong\n");
    }

    printf("reduce_baseline latency = %f ms\n", milliseconds);

// 释放CPU端数据
    free(cpu_arr);
    free(cpu_sum);
    free(cpu_right_sum);

// 释放GPU端数据
    cudaFree(gpu_arr);
    cudaFree(gpu_sum);

// 释放计时数据
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
