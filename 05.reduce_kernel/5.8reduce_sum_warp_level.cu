//相较于v1-v6在block层面实现归约求和，本次是在warp层面实现

/*
总结：
在warp层面相较于block层面的好处：1.线程之间的同步开销相较blcok的__syncthreads();小很多。2.不用考虑是否存在warp divergence。3.warp层面有更高的通信带宽。
但是，不一定在warp层面就一定会有显著提升，要case by case，例如数据量大的情况下，warp层面的reduce可能效率不如block层面。

warp层面相比block层面需要拆分成两次归约，第一次是warp内部归约，第二次是拆分的多个warp归约。

*/


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
__global__ void reduce_warp_level(float *gpu_arr,float *gpu_sum, unsigned int N)
{

    float sum = 0;//当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器
    unsigned int local_block_threadidx  = threadIdx.x;
    unsigned int global_block_threadidx  = blockIdx.x * blockDim.x + threadIdx.x;

    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过for循环来自动确定每个线程处理的元素个数
    for (int index = global_block_threadidx ; index < N; index += blockDim.x * gridDim.x)  // 在启动kernel时，并未减少线程，因此，此处依然是1个线程对应1个数据
    {
        sum += gpu_arr[index];   // 每个线程只会执行1次，因此每个线程对应的sum应该都为1
    }


    // 与block层面对每个线程存结果不同的是，warp层面是对1个block中的每个warp的和存为一个结果，即有多少个warp存多少个结果，blockSize=256，warp=32，即一个block存8个warp
    __shared__ float shared_WarpSums[blockSize / WarpSize];   // 范围：WarpSums[0-8]

    // 当前线程在其所在warp内的ID，例如threads=129，它在block层面就是129，但是在warp层面，它就是第129/32=4……1 = 5个warp中的第1个线程
    const int laneId = local_block_threadidx % WarpSize;
    // 当前线程所在warp在所有warp范围内的ID，例如threads=129，它在block层面就是blockDim.x*blockIdx.x+threadIdx.x，但是在warp层面，它就是第129/32=4……1 = 5个warp中
    const int warpId = local_block_threadidx / WarpSize; 


    

    // 第一次归约求和：对当前线程所在warp做warpshuffle操作，求得一个warp的和
    sum = WarpShuffle<blockSize>(sum);

    if(laneId == 0) 
    {
        shared_WarpSums[warpId] = sum;  // 将1个warp的和存到WarpSums中，一共8个
    }
    __syncthreads();


    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和，也就是将WarpSums中的8个结果再次求和
    sum = (local_block_threadidx < blockSize / WarpSize) ? shared_WarpSums[laneId] : 0;   // 这里将前8个线程变为warp和，其余置零，否则会污染计算结果。不置零则如下：[32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    
    if (warpId == 0) 
    {
        sum = WarpShuffle<blockSize/WarpSize>(sum);   // 第二次归约求8个warp组成的1个block结果
    }
    
    if (local_block_threadidx == 0)  
    {
        gpu_sum[blockIdx.x] = sum;   // 得到第二次归约后1个block的结果
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
