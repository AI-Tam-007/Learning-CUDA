/*
总结：

1.决定block大小看：
                    maxThreadsPerBlock
                    warpSize
                    regsPerBlock
                    sharedMemPerBlock
一般选择：128 / 256 / 512

2.决定 shared memory 用量看：
                              sharedMemPerBlock
                              sharedMemPerMultiprocessor
否则block 上不去 → occupancy 低。

3.决定kernel并发度看：
                      maxThreadsPerMultiProcessor
                      regsPerBlock
                      sharedMemPerMultiprocessor

能够直观确定每个SM能跑几个block。

4.结合带宽测试程序：
                      情况	                 说明
                     带宽低	               访存瓶颈
                     计算慢	               算力瓶颈
                   occupancy低	           资源瓶颈

*/



#include <cuda_runtime.h>    // 提供cudaGetDeviceCount、cudaGetDeviceProperties、cudaSetDevice等API
#include <cuda.h>
#include <stdio.h>
#include <string>

using namespace std;
int main() 
{


    int deviceCount = 0;   // 初始化一个变量,用于存储机器中有几个可用的CUDA显卡的数量
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);  // 获取当前机器可用的GPU数量, cudaError_t实际上是一个枚举类型，它所包含的每一个值都对应着CUDA操作可能出现的一种状态。
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount failed: %s\n",cudaGetErrorString(error_id));
        return -1;
    }


    if (deviceCount == 0)    // 如果deviceCount数量等于0，说明没有可用的gpu
    {
        printf("There are no available device(s) that support CUDA\n");
        return 0;
    } 
    else 
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }


    for (int dev = 0; dev < deviceCount; ++dev)    // 从0号GPU开始，遍历每一张GPU
    {
        cudaSetDevice(dev);   // 使用指定的GPU，例如0号GPU
        cudaDeviceProp deviceProp;  // 初始化当前device的属性获取对象
        cudaGetDeviceProperties(&deviceProp, dev);  // 把GPU设备编号dev的详细属性填充到deviceProp结构体中

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);   // 打印GPU设备编号dev和设备名称
        
        // 打印全局显存总量，其直接影响能处理多大规模的问题，总字节数deviceProp.totalGlobalMem除以1048576.0f的含义是将字节B转换为MB
        printf("  Total amount of global memory:                 %.0f MBytes " "(%llu bytes)\n", static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),(unsigned long long)deviceProp.totalGlobalMem);


        // 打印GPU最大时钟频率（每个SM的频率）---原始单位是kHz → 转换为MHz和 GHz，含义是GPU算术核心能够跑多块，类似CPU主频，该参数对计算密集型kernel影响大。
        printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f " "GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        // 打印L2缓存(Cache)大小（全GPU共享缓存，减少对全局内存的访问，对需要重复访问的数据很重要）
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        
        // ★：每个block中可用的Shared Memory总量（通常48k或96k），直接影响occupancy，occupancy指的是一个SM上实际运行的warp数/理论最大warp数。
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);

        // ★：一个SM中的所有block的共享内存总上限，总量越大，一个SM能驻留更多block，并发度越高。
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);

        // ★：每个Block可使用的寄存器数量，regsPerThread(每个线程使用的寄存器数量) × threadsPerBlock(每个block中启动的线程数量=blockSize) ≤ regsPerBlock(硬件允许一个block最多使用的寄存器总数)
        // ⚠：★一旦超出会降低occupancy，对性能影响极大。★
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);

        // ★：Warp大小：通常固定是 32线程，1个Warp是GPU执行指令最小的调度单位
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);

        // ★：每个SM最多能同时容纳的线程数，一般为2048，对应64个Warp，决定occupancy上限。
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);

        // ★：每个Block中线程数的上限，一般为1024
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

        // ★：每个Block的三维尺寸限制，通常情况：block.x ≤ 1024,block.y ≤ 1024,block.z ≤ 64并且三者总乘积≤ maxThreadsPerBlock，否则kernel 启动会失败，CUDA 运行时返回错误。
        printf("  Max dimension size of a block size(x,y,z):     (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        // ★：整个Grid的三维尺寸限制，通常情况：x方向极大（2^31-1）,y/z较小
        printf("  Max dimension size of a grid size(x,y,z):      (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    }

    return 0;
}
