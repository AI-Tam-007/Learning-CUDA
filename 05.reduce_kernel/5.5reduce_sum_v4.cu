// 相较v3而言，v4的改进点主要是由于__syncthreads()开销还是比较大的，所以将for循环中的最后一个warp拎出来，做__syncwarp()。


/*
总结：
1.最后一个warp在展开时要注意数据是否同步的问题。

*/


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



template<int blockSize>
__global__ void reduce_v4(float *gpu_arr, float *gpu_sum, int N)
{
  
    __shared__ float shared_memory[blockSize];   // 注意：kernel内部的blockSize并非外部的blockSize值，因为外部在启动kernel时用的是blockSize/2
    unsigned int local_block_threadidx = threadIdx.x;
    unsigned int global_block_threadidx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;  // 此处*2指仍要处理2*blockSize的数据，即256个，只是线程数减成128而已

    
    // 每个线程加载两个元素到shared mem对应位置（边界保护）
    float sum = 0.0f;
    if (global_block_threadidx < N) sum += gpu_arr[global_block_threadidx];
    if (global_block_threadidx + blockSize < N) sum += gpu_arr[global_block_threadidx + blockSize];

    shared_memory[local_block_threadidx] = sum;
  
    __syncthreads();
    // 执行到此处后，数据量从N到了N/2
    
    
    // 相较v3，此处循环执行到最后32个线程即可结束
    for (unsigned int index = blockDim.x / 2; index > 32; index >>= 1) 
    {
        if (local_block_threadidx < index) 
        {
            shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + index];
        }
        __syncthreads();
    }


    // 最后一个warp拎出来单独作reduce
    if (local_block_threadidx < 32) 
    {
        WarpSharedMemReduce(shared_memory, local_block_threadidx);
    }


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
    dim3 block(blockSize / 2);

    // 设定grid大小
    int gridSize = min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);

    // host端定义变量并分配内存
    float *cpu_arr = (float *)malloc(N * sizeof(float));
    float *cpu_sum = (float*)malloc((gridSize) * sizeof(float));

    // host端数据初始化
    for(int i = 0; i < N; i++){
        cpu_arr[i] = 1.0f;
    }



    // device端定义变量并分配内存
    float *gpu_arr;  
    float *gpu_sum;
    cudaMalloc((void **)&gpu_arr, N * sizeof(float));
    cudaMalloc((void **)&gpu_sum, (gridSize) * sizeof(float));
    
    // 用于验证device端输出结果的正确性
    float groundtruth = N * 1.0f;

    // 将host端的cpu_arr数据转到device端的gpu_arr数据中
    cudaMemcpy(gpu_arr, cpu_arr, N * sizeof(float), cudaMemcpyHostToDevice);


    //  ==============热身===================
    reduce_v4<blockSize / 2><<<grid,block>>>(gpu_arr, gpu_sum, N);  // 热身，为了计算更准确的时间
    cudaDeviceSynchronize();
    // =====================================


    // 记录核函数处理时间
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动模板kernel函数，blockSize / 2表示只用一半，原来是256，现在只用128
    reduce_v4<blockSize / 2><<<grid,block>>>(gpu_arr, gpu_sum, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将device端的gpu_sum数据转到host端的cpu_sum数据中
    cudaMemcpy(cpu_sum, gpu_sum, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
  
    // 已分配gridSize个块，总数据量为N
    cout << "allcated " << gridSize << " blocks, data counts are " << N << endl;

    // 撰写函数检验cpu端数据和gpu端数据是否一致
    bool is_right = CheckResult(cpu_sum, groundtruth, gridSize);
    if(is_right) 
    {
        cout << "the ans is right" << endl;
    } 
    else 
    {
        cout << "the ans is wrong" << endl;
        cout << "groundtruth is:" << groundtruth << endl;
        for(int i = 0; i < gridSize; i++)
        {
            cout << "res per block :" << cpu_sum[i] << endl;
        }
    }
    cout << "reduce_v4 latency :" << milliseconds << endl;

    // 释放资源
    cudaFree(gpu_arr);
    cudaFree(gpu_sum);
    free(cpu_arr);
    free(cpu_sum);
}
