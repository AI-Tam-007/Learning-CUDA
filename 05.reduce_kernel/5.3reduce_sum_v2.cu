// 相较v1而言，v2的改进点主要是将 x % k 转化为 x & (k-1)


/*
总结：
1.'%'非常耗时,将 x % k 转化为 x & (k-1)后耗时显著提升，但是优化%运算符的前提条件是k必须是2的幂次，这个时候才能将 x % k 转化为 x & (k-1)。
2.我们写的kernel并未产生bank conflict，因此补充一个bank conflict示例如下：


            for (int i = 1; i < blockDim.x; i *= 2) 
            {
                int index = 2 * i * threadIdx.x;
                if (index < blockDim.x) {
                    shared_memory[index] += smem[index + i];
                }
                __syncthreads();
            }

            
bank conflict推导：
执行for (int i = 1; i < blockDim.x; i *= 2)这个指令，当i = 1时，某一个block中的256个线程并行执行i=1时for循环中的内容，假设线程从0-255执行，index的范围为0,2,4,6……560；
由于index < blockDim.x，所以，只有0，2，4，6……254满足条件，也就是编号为0-127的线程，此时，0号线程对应的bank=1，1号线程对应的bank=3，同理15号线程对应着bank = 31，
假设bank是从0-31的序号，则15号线程对应的是最后一个bank，16号线程则对应着bank = 33，实际为bank=1，冲突了，17号线程bank = 35，实际是bank = 3也冲突，同理。
注意：bank conflict是存在与shared memory中特有的，如果是在全局内存中，则是多线程访存冲突，如果是逻辑层面的 “竞态条件”（数据竞争），应该使用原子函数atomics，如果是硬件层面的 “访存合并失效”应该优化访存模式（让线程连续、对齐访问）。

*/



#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;


template<int blockSize>    // 可用template<int blockSize>和#define  blockSize 256，但template<int blockSize>是最优选择
__global__ void sum(int *gpu_arr, int *gpu_sum, int N)
{
    // 每个block内部都有自己的share memory，shared memory延迟更低，带宽更高，因此在share memory中实现速度要快些。
    __shared__ int share_memory[blockSize];
    int local_block_threadidx = threadIdx.x;   // 在某一个block中的thread的id
    int global_block_threadidx = blockDim.x * blockIdx.x + threadIdx.x;  // 在全局block中的thread的id

    
    // 将每个线程对应的数据加载到share memory中，如果N/block不为整数，则此处最后一个block存在warp divergence，但是影响不大，相对整个N很小
    if (global_block_threadidx < N)  // 如果没有这句话，当线程数大于数据数量时，数组可能会越界。
    {

        // 为什么这样写，因为不同的block中的thread的索引都是从0开始，正好对应着local_block_threadidx。
        // 所以，当shared memory改变后，应该和local_block_threadidx一样继续从0开始。
        // 但是gpu_arr的数据可不一样，不同的block是接在一起的，用全局线程索引就能将所有数据都串在一起，
        shared_memory[local_block_threadidx] = gpu_arr[global_block_threadidx];
    }
    else
    {
        shared_memory[local_block_threadidx] = 0;   // 这里为什么要填0，避免后续计算将“垃圾值”加载到数据中了。
    }
    __syncthreads();   // 保证别的线程读之前都写完，先确保写完再去读取，这样避免读取到旧值，错值。


    
    // 这里为什么是blockDim.x，因为local_block_threadidx是某block中的一个thread的id，而blockDim.x就是某一个block的大小，但不一定就是定值256。
    for(int index = 1; index < blockDim.x; index *= 2)
    {
      
       
        // ⚠v2相较v1优化此处，将'%'修改为'&'
        if((local_block_threadidx & (2 * index - 1)) == 0 && (2 * index) <= blockDim.x)  // 个人认为此处并未出现warp divergence，else为空，影响可忽略不计，实测后的效率也并没有什么影响。
        {
            shared_memory[local_block_threadidx] += shared_memory[local_block_threadidx + index];
        }
        __syncthreads();
    }

    

    // 这里为什么要有这个判断语句？因为如果没有这个判断语句，100000个block中每一个线程都会并行执行这个语句，导致产生竟态条件。
    // 并且让local_block_threadidx == 0是约定俗成的，也可以不== 0，由于最终归约的结果是在[0]号位，最好是用0，方便易理解。
    if(local_block_threadidx == 0)
    {
        gpu_sum[blockIdx.x] = shared_memory[0];
    }

}



bool CheckResult(int gpu_right_sum, int cpu_right_sum) 
{
    // int类型的情况
    return (gpu_right_sum == cpu_right_sum);

    // float类型的情况
    // return fabs(*cpu_sum - cpu_right_sum) <= (1e-6f) * fmax(1.0f, fmax(fabs(*cpu_sum), fabs(cpu_right_sum)));

    // double类型的情况
    // return fabs(*cpu_sum - cpu_right_sum) <= (1e-12) * fmax(1.0f, fmax(fabs(*cpu_sum), fabs(cpu_right_sum)));
}



int main()
{

    int N = 25600000;
    /*定义GPU的grid和block*/
    int blockSize = 256;   // 定义一个block大小，即blockDim.x大小，也就是1个block中thread的数量，最好设置为32的倍数并且是2的幂次，否则在核函数中容易发生越界的可能
    int gridSize = (N + blockSize - 1) / blockSize;   // 定义一个grid大小，即gridDim.x，也就是block数量。如果是一维，即表示gridSize个block，如果是二维，即表示gridSize*gridSize个block
    dim3 block(blockSize);  // 一维，即一个block中包含blockSize个线程
    dim3 grid(gridSize);   // 一维，即一个grid中包含gridSize个block



    cudaSetDevice(0);    // 指定使用的CUDA设备
    cudaDeviceProp deviceProp;  // 存储 GPU 设备的各种属性信息
    cudaGetDeviceProperties(&deviceProp, 0);   // 把GPU设备0的详细属性填充到deviceProp结构体中
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", 0, deviceProp.name, deviceProp.major, deviceProp.minor);


    int nbytes = N * sizeof(int);   // 25600000大小空间
    int nbytes_block = grid_num * sizeof(int);  // 这里为什么是grid_num？因为只归约1次，剩余的数据量就是grid_num=100000个，后续要根据数据量大小判断是否需要继续归约，数据量依然大可以继续归约，数据量小可以拷贝到CPU端直接求和。


    // 在堆中分配内存
    int *cpu_arr = (int *)malloc(nbytes);   // cpu(host)端分配内存---系统内存
    int *cpu_sum = (int *)malloc(nbytes_block);  // cpu(host)端分配内存---系统内存

    int cpu_right_sum = 0;
    

    // cpu初始化并计算数组和
    for(int i = 0; i < N; i++)
    {
        cpu_arr[i] = 1;    //等价于*(cpu_arr + i)
        cpu_right_sum += cpu_arr[i];  // *cpu_right_sum等价于cpu_right_sum[0],为了确保可读性，最好是用*cpu_right_sum代表单个值的数组
    }




     // 在显存中分配内存
    int *gpu_arr, *gpu_sum; 
    cudaMalloc((void **)&gpu_arr, nbytes);      // gpu(device)端分配内存---系统显存
    cudaMalloc((void **)&gpu_sum, nbytes_block);  // gpu(device)端分配内存---系统显存
    int gpu_right_sum = 0;
    

    
// 将CPU端的数据拷贝到GPU端
    cudaMemcpy(gpu_arr, cpu_arr, nbytes, cudaMemcpyHostToDevice);  // 将cpu(host)端的数据转到gpu(device)端



    //  ==============热身===================
    sum<<<grid, block>>>(gpu_arr, gpu_sum, N);   // 热身，为了计算更准确的时间
    cudaDeviceSynchronize();
    // =====================================



// 计算核函数运行时间
    const int iterations = 100;  // 迭代100次核函数，计算更精确的gpu运行时间
    float milliseconds = 0;     // 计算gpu运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


// 启动核函数
    for (int i = 0; i < iterations; i++)    // 执行100次，为了计算更精确的时间
    {
        sum<<<grid, block>>>(gpu_arr, gpu_sum, N); 
    }  



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / iterations;

// 将GPU端的结果转移到CPU端
    cudaMemcpy(cpu_sum, gpu_sum, nbytes_block, cudaMemcpyDeviceToHost);


    // 此处并未执行第二次归约，而是直接将结果拷贝到CPU端进行加和。
    for (int i = 0; i < grid_num; i++) // 计算100000个block的和，至于归约到多少个数据后能在cpu中求和，得看数据还有复杂度
    {
        gpu_right_sum += cpu_sum[i];
    }
    printf("cpu_right_sum:%d \n", cpu_right_sum);
    printf("gpu_right_sum:%d \n", gpu_right_sum);

    printf("reduce_baseline latency = %f ms\n", avg_ms);


// 验证CPU和GPU计算出的数据是否一致
    bool is_right = CheckResult(gpu_right_sum, cpu_right_sum);
    if(is_right) 
    {
        printf("the ans is right\n");
    } 
    else 
    {
        printf("the ans is wrong\n");
    }

    

// 释放CPU端数据
    free(cpu_arr);
    free(cpu_sum);

// 释放GPU端数据
    cudaFree(gpu_arr);
    cudaFree(gpu_sum);

// 释放计时数据
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

