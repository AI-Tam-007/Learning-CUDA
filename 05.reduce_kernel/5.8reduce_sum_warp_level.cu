//相较于v1-v6在block层面实现归约求和，本次是在warp层面实现




#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define WarpSize 32   // 宏定义
 
//latency: 1.254ms
template <int blockSize>
__device__ float WarpShuffle(float sum) 
{
    // 1.__shfl_down_sync：前面的thread向后面的thread要数据，__shfl_down_sync(0xffffffff, sum, 16)意思就是获得sum线程向后偏移16个线程的数据
    // 2.__shfl_up_sync: 后面的thread向前面的thread要数据
    // 3. 使用warp shuffle指令的数据交换不会出现warp在shared memory上交换数据时的不一致现象，这一点是由GPU driver完成，故无需任何sync, 比如syncwarp
    // 4. 原先15-19行有5个if判断block size的大小，目前已经被移除，确认了一下__shfl_down_sync等warp shuffle指令可以handle一个block或一个warp的线程数量<32，不足32会自动填充0
    sum += __shfl_down_sync(0xffffffff, sum, 16);    // 0-16, 1-17, 2-18, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 8);     // 0-8, 1-9, 2-10, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 4);     // 0-4, 1-5, 2-6, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 2);     // 0-2, 1-3, 4-6, 5-7, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 1);     // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *gpu_arr,float *gpu_sum, unsigned int N){
    float sum = 0;//当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    unsigned int local_block_threadidx  = threadIdx.x;
    unsigned int global_block_threadidx  = blockIdx.x * blockDim.x + threadIdx.x;

    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L30的for循环来自动确定每个线程处理的元素个数
    for (int index = global_block_threadidx ; index < N; index += blockDim.x * gridDim.x)
    {
        sum += gpu_arr[index];
    }
    
    // 用于存储partial sums for each warp of a block
    __shared__ float WarpSums[blockSize / WarpSize]; 
    // 当前线程在其所在warp内的ID
    const int laneId = local_block_threadidx % WarpSize;
    // 当前线程所在warp在所有warp范围内的ID
    const int warpId = local_block_threadidx / WarpSize; 
    // 对当前线程所在warp作warpshuffle操作，直接交换warp内线程间的寄存器数据
    sum = WarpShuffle<blockSize>(sum);
    if(laneId == 0) 
    {
        WarpSums[warpId] = sum;
    }
    __syncthreads();
    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    //首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    //接着，继续warpshuffle
    sum = (local_block_threadidx < blockSize / WarpSize) ? WarpSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) 
    {
        sum = WarpShuffle<blockSize/WarpSize>(sum); 
    }
    // store: 哪里来回哪里去，把reduce结果写回显存
    if (local_block_threadidx == 0) 
    {
        gpu_sum[blockIdx.x] = sum;
    }
}

bool CheckResult(float *cpu_sum, float groudtruth, int gridSize)
{
    float res = 0;
    for (int i = 0; i < gridSize; i++)
    {
        res += cpu_sum[i];
    }
    if (res != groudtruth) 
    {
        return false;
    }
    return true;
}

int main()
{
    
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256;
    dim3 block(blockSize);

    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);

    //int GridSize = 100000;
    float *cpu_arr = (float *)malloc(N * sizeof(float));
    float *cpu_sum = (float*)malloc((gridSize) * sizeof(float));

    float *gpu_arr;
    float *gpu_sum;
    cudaMalloc((void **)&gpu_arr, N * sizeof(float));
    cudaMalloc((void **)&gpu_sum, (gridSize) * sizeof(float));


    for(int i = 0; i < N; i++)
    {
        cpu_arr[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(gpu_arr, cpu_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    
    

    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<grid,block>>>(gpu_arr, gpu_sum, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(cpu_sum, gpu_sum, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", gridSize, N);
    bool is_right = CheckResult(cpu_sum, groudtruth, gridSize);
    if(is_right) 
    {
        printf("the ans is right\n");
    } 
    else 
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < gridSize;i++)
        {
            printf("resPerBlock : %lf ",cpu_sum[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    cudaFree(gpu_arr);
    cudaFree(gpu_sum);
    free(cpu_arr);
    free(cpu_sum);
}
