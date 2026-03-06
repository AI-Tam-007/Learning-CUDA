// 相较v2而言，v3的改进点主要是把一部分本来要在 shared memory 里做的第一次加法，提前到 global memory load 阶段完成。
// 这样每个线程在进入归约前先做了更多有效工作，从而提升线程利用率，并减少 shared memory 内参与归约的数据量。
//（特别注意：v3只是减少block中的线程数量，从256减少到128，而并不是减少block个数，虽然减少到128个线程，但是1个block仍然是处理256个元素。）



/*
个人理解：（在v2设定block和grid数量时是按照一个线程对应一个数据，经过第一轮归约后，有一半的线程不工作了，随着归约不断进行，越往后会让越来越多的线程空闲下来，因此思路就是解决一开始空闲线程过多的问题）

解决原理：假设原本是1024个数要计算，每个block包含256个线程，因此需要4个block，每个数对应1个thread，每个线程将结果存到shared memory中，进入第一轮归约后，有一半的线程空闲，
          当我们只设定每个block只有原先一半线程时(128)，然后在存到shared memory的过程中先一步进行求和，这样进入第一轮归约后，只有四分之一（64）的线程空闲，相比之前二分之一(128)的空闲来说，算是有效提升。
          
          (原来每个 block 用 256 个线程处理 256 个数，第一轮 shared memory 归约时有 128 个线程空闲。现在每个 block 只用 128 个线程，但每个线程先处理两个数，因此在进入 shared memory 归约
          前就完成了一轮局部求和。所以虽然归约阶段每一轮仍然是“一半线程活跃、一半线程空闲”，但线程在空闲之前先完成了更多实际工作，整体线程利用率更高。)
*/


#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>

using namespace std;



template<int blockSize>
__global__ void reduce_v3(float *gpu_arr, float *gpu_sum, int N)
{
  
    __shared__ float shared_memory[blockSize];   // 注意：kernel内部的blockSize并非外部的blockSize值，因为外部在启动kernel时用的是blockSize/2
    unsigned int local_block_threadidx = threadIdx.x;
    unsigned int global_block_threadidx = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // 此处*2指仍要处理2*blockSize的数据，即256个，只是线程数减成128而已

    
    // 每个线程加载两个元素到shared mem对应位置（边界保护）
    float sum = 0.0f;
    if (global_block_threadidx < N) sum += gpu_arr[global_block_threadidx];
    if (global_block_threadidx + blockSize < N) sum += gpu_arr[global_block_threadidx + blockSize];

    shared_memory[local_block_threadidx] = sum;
  
    __syncthreads();
    // 执行到此处后，数据量从N到了N/2
    
    
    // 开始对每个block进行归约求和
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) 
    {
        if (local_block_threadidx < index) 
        {
            shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + index];
        }
        __syncthreads();
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

    
    // 记录核函数处理时间
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 启动模板kernel函数，blockSize / 2表示只用一半，原来是256，现在只用128
    reduce_v3<blockSize / 2><<<grid,block>>>(gpu_arr, gpu_sum, N);

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
    cout << "reduce_v3 latency :" << milliseconds << endl;

    // 释放资源
    cudaFree(gpu_arr);
    cudaFree(gpu_sum);
    free(cpu_arr);
    free(cpu_sum);
}
