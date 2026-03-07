// 相较v5而言，v6的改进点主要是启动两次核函数求得最终完整的结果，前面几个版本只启动了一次kernel，然后将算出的gridSize个block的和在host端进行加和。



#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>

using namespace std;


__device__ void WarpSharedMemReduce(volatile float* shared_memory, int local_block_threadidx)
{
    
    // 这里为什么需要一个中间变量temp作为媒介？不能直接用shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + 32]…………？
    // 解释：因为Volta架构引入的独立线程调度导致的，也就是每个线程具有独立的PC和栈S，无法保证每个线程处于同一水平面执行，导致有快有慢，因此用一个中间变量将读和写分开，
    // 再配合上volatile（告诉编译器shared memory的值随时可能被别的线程改掉；每次用它都要真的去内存里读/写）和__syncwarp();（让 warp 内线程执行同步），保证了结果的准确性。
    float temp = shared_memory[local_block_threadidx];
    if (blockDim.x >= 64) 
    {
      temp += shared_memory[local_block_threadidx + 32]; __syncwarp();
      shared_memory[local_block_threadidx] = temp; __syncwarp();
    }

    // 这里是因为在同一个warp内，所以不必添加判断语句
    temp += shared_memory[local_block_threadidx + 16]; __syncwarp();
    shared_memory[local_block_threadidx] = temp; __syncwarp();

    temp += shared_memory[local_block_threadidx + 8]; __syncwarp();
    shared_memory[local_block_threadidx] = temp; __syncwarp();

    temp += shared_memory[local_block_threadidx + 4]; __syncwarp();
    shared_memory[local_block_threadidx] = temp; __syncwarp();

    temp += shared_memory[local_block_threadidx + 2]; __syncwarp();
    shared_memory[local_block_threadidx] = temp; __syncwarp();

    temp += shared_memory[local_block_threadidx + 1]; __syncwarp();
    shared_memory[local_block_threadidx] = temp; __syncwarp();

}



template <int blockSize>
__device__ void BlockSharedMemReduce(float* shared_memory, int local_block_threadidx) 
{
    //对v4 L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
    if (blockSize >= 1024)
    {
        if (local_block_threadidx < 512) 
        {
          shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + 512];
        }
        __syncthreads();
    }

    if (blockSize >= 512) 
    {
        if (local_block_threadidx < 256) 
        {
          shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) 
    {
        if (local_block_threadidx < 128) 
        {
          shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) 
    {
        if (local_block_threadidx < 64) 
        {
          shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + 64];
        }
        __syncthreads();
    }
    // the final warp
    if (local_block_threadidx < 32) 
    {
        WarpSharedMemReduce(shared_memory, local_block_threadidx);
    }
}




template<int blockSize>
__global__ void reduce_v6(float *gpu_arr, float *gpu_sum, int N)
{
  
    __shared__ float shared_memory[blockSize]; 
    unsigned int local_block_threadidx = threadIdx.x;
    unsigned int global_block_threadidx = blockIdx.x * blockDim.x + threadIdx.x;  

    
    
    
    float sum = 0.0f;

    // 此处的步长为全部线程总数，可能并未显现出来，但是当blockDim.x*gridDim.x远小于数据量的时候grid-stride loop就会起作用
    for (unsigned int index = global_block_threadidx ; index < N; index += blockDim.x*gridDim.x)    // 不一定会提升性能
    {
        sum += gpu_arr[index];
    }
    shared_memory[local_block_threadidx] = sum;
    __syncthreads();



    // 相较v4，将此处for循环完整展开并且最后一个warp依然被拎出来单独作reduce
    BlockSharedMemReduce<blockSize>(shared_memory, local_block_threadidx);


    
    // gridSize个block的reduce sum已得出，保存到gpu_sum中，供后续继续归约使用
    if (local_block_threadidx == 0) 
    {
        gpu_sum[blockIdx.x] = shared_memory[0];
    }


}

bool CheckResult(float *cpu_sum, float groundtruth, int gridSize)
{
    float res = 0;
    for (int i = 0; i < gridSize; i++)
    {
        res += cpu_sum[i];
    }
    if (fabs(res - groundtruth) > 1e-3) 
    {
        return false;
    }
    return true;
}

int main()
{

    const int N = 25600000;   // 假设25600000万个数据量
  
    cudaSetDevice(0);      // 指定使用0号GPU
    cudaDeviceProp deviceProp;   // 定义deviceProp用于存储GPU设备的硬件参数
    cudaGetDeviceProperties(&deviceProp, 0);    // 获取0号GPU的所有硬件属性，并将结果填充到deviceProp中

    // 设定block大小，相较v2，减少一半线程
    const int blockSize = 256;   
    dim3 block(blockSize);

    // 设定grid大小
    int gridSize = min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);

    // host端定义变量并分配内存
    float *cpu_arr = (float *)malloc(N * sizeof(float));
    float *cpu_sum = (float *)malloc(1 * sizeof(float));
    

    // host端数据初始化
    for(int i = 0; i < N; i++)
    {
        cpu_arr[i] = 1.0f;
    }



    // device端定义变量并分配内存
    float *gpu_arr;  
    float *gpu_sum;
    cudaMalloc((void **)&gpu_arr, N * sizeof(float));
    cudaMalloc((void **)&gpu_sum, (gridSize) * sizeof(float));


    // 新增变量
    float *all_result;
    cudaMalloc((void **)&all_result, 1 * sizeof(float));


    // 用于验证device端输出结果的正确性
    float groundtruth = N * 1.0f;

    // 将host端的cpu_arr数据转到device端的gpu_arr数据中
    cudaMemcpy(gpu_arr, cpu_arr, N * sizeof(float), cudaMemcpyHostToDevice);


    //  ==============热身===================
    reduce_v6<blockSize><<<grid,block>>>(gpu_arr, gpu_sum, N);  // 热身，为了计算更准确的时间
    cudaDeviceSynchronize();
    // =====================================


    // 记录核函数处理时间
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动模板kernel函数
    reduce_v6<blockSize><<<grid,block>>>(gpu_arr, gpu_sum, N);
    reduce_v6<blockSize><<<1,block>>>(gpu_sum, all_result, gridSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将device端的gpu_sum数据转到host端的cpu_sum数据中
    cudaMemcpy(cpu_sum, all_result, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  
    // 已分配gridSize个块，总数据量为N
    cout << "allcated " << gridSize << " blocks, data counts are " << N << endl;

    // 撰写函数检验cpu端数据和gpu端数据是否一致
    bool is_right = CheckResult(cpu_sum, groundtruth, 1);
    if(is_right) 
    {
        cout << "the ans is right" << endl;
    } 
    else 
    {
        cout << "the ans is wrong" << endl;
        cout << "groundtruth is:" << groundtruth << endl;
        for(int i = 0; i < 1; i++)
        {
            cout << "res per block :" << cpu_sum[i] << endl;
        }
    }
    cout << "reduce_v6 latency :" << milliseconds << endl;

    // 释放资源
    cudaFree(gpu_arr);
    cudaFree(gpu_sum);
    cudaFree(all_result);
    free(cpu_arr);
    free(cpu_sum);
}
