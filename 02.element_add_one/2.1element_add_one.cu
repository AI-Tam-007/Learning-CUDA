/*
====== 数组值加1 ======
总结：
1. 在cpu(cpu_arr)中分配内存：cpu_arr = (float *)malloc(N*sizeof(float));
2. 在cpu(result_arr)中分配内存：result_arr = (float *)malloc(N*sizeof(float));
3. 在gpu(gpu_arr)中分配内存：cudaMalloc((void **)&gpu_arr, N*sizeof(float)); 
4. 从cpu(cpu_arr)-gpu(gpu_arr)：cudaMemcpy(gpu_arr, cpu_arr, N*sizeof(float), cudaMemcpyHostToDevice);
5. 从gpu(gpu_arr)-cpu(result_arr)：cudaMemcpy(result_arr, gpu_arr, N*sizeof(float), cudaMemcpyDeviceToHost);
6. 释放cpu(cpu_arr)分配的内存：free(cpu_arr);
7. 释放cpu(result_arr)分配的内存：free(result_arr);
8. 释放gpu分配的内存：cudaFree(gpu_arr);

*/

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>   

__global__ void add_kernel(float *gpu_arr)   // 核函数，传入一个可变的值
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;   // 计算每个线程的唯一索引

    gpu_arr[global_idx] += 1;   // 对传入到gpu中的每个数据的值都加1

}


int main()
{
    int N = 32;   // 总共有32个数
    int blockSize = 32;  // 1个block假设只有32个线程
    float *cpu_arr, *gpu_arr, *result_arr;   // 定义cpu、gpu和result数组

    cpu_arr = (float *)malloc(N*sizeof(float));  // 给cpu_arr分配N*sizeof(float)内存

    result_arr = (float *)malloc(N*sizeof(float));  // 给result_arr分配N*sizeof(float)内存

    cudaMalloc((void **)&gpu_arr, N*sizeof(float));  // ★给gpu_arr分配N*sizeof(float)内存。




    printf("cpu_arr:\n");     // 打印cpu初始化的数据
    for(int i = 0; i < N; i++)
    {
        cpu_arr[i] = i;
        printf("%g  ",cpu_arr[i]);
    }


    cudaMemcpy(gpu_arr, cpu_arr, N*sizeof(float), cudaMemcpyHostToDevice);   // 将cpu的数据拷贝到gpu上

    add_kernel<<<1,blockSize>>>(gpu_arr);  // 启动核函数， 一个block，32个线程
 
    cudaMemcpy(result_arr, gpu_arr, N*sizeof(float), cudaMemcpyDeviceToHost);  // 将gpu处理后的数据拷贝到result中

    printf("\nresult_arr:\n");  // 打印gpu处理完的数据result
    for(int i = 0; i < N; i++)
    {
        printf("%g  ", result_arr[i]);
    }

    free(cpu_arr);    // 释放cpu(cpu_arr)分配的内存
    free(result_arr); // 释放cpu(result_arr)分配的内存
    cudaFree(gpu_arr);  // 释放gpu(gpu_arr)分配的内存

    return 0;
}
