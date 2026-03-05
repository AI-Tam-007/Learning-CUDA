/*
=======设备查询(获取GPU详细参数)=======
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1060"
  Total amount of global memory:                 6144 MBytes (6442319872 bytes)   // 总显存大小， GPU的全局DRAM容量，供所有线程块共享访问。对于数据集较大的应用非常关键。
  GPU Max Clock rate:                            1733 MHz (1.73 GHz)   // GPU核心的运行频率，更高的时钟频率通常意味着更快的执行速度，但也可能导致更高功耗。
  L2 Cache Size:                                 1572864 bytes     // L2缓存是所有SM共享的高速缓存，能显著减少访问全局内存时的延迟。
  Total amount of shared memory per block:       49152 bytes       // 一个block(线程块)中最多能用的shared memory(共享内存)的数量。
  Total shared memory per multiprocessor:        98304 bytes       // 一个SM(相当于CPU中的核)可分配给多个线程块的总共享内存。
  Total number of registers available per block: 65536             // 一个block中最多可用的寄存器总数，CUDA会自动按照线程分配。
  Warp size:                                     32                // 一个warp线程的大小，同一个warp内线程同步执行。
  Maximum number of threads per multiprocessor:  2048              // 一个SM同时可以托管的活跃线程数峰值，通常对应64warp。
  Maximum number of threads per block:           1024              // 一个block中所支持的最大线程数量。
  Max dimension size of a block size (x,y,z): (1024, 1024, 64)     // 一个block在 x、y、z 方向上的最大尺寸。三个维度的总乘积不能超过上面的block最大线程数量的限制。
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)   // 一个grid在 x、y、z 方向上的最大尺寸。它决定了kernel launch时grid的大小。

*/



#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>

int main() 
{


    int deviceCount = 0;   // 初始化一个变量
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);  // 获取当前机器的GPU数量, cudaError_t实际上是一个枚举类型，它所包含的每一个值都对应着CUDA操作可能出现的一种状态。
    if (deviceCount == 0)   // 如果deviceCount数量等于0，说明没有可用的gpu
    {
        printf("There are no available device(s) that support CUDA\n");
        return 0;
    } 
    else 
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)   // 遍历每一张GPU
    {
        cudaSetDevice(dev);   // 使用指定的GPU
        cudaDeviceProp deviceProp;  // 初始化当前device的属性获取对象
        cudaGetDeviceProperties(&deviceProp, dev);  // 把GPU设备编号dev的详细属性填充到deviceProp结构体中

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);   // 打印GPU设备编号dev和设备名称
        
        // 打印显存总量（全局内存-Global Memory），总字节数deviceProp.totalGlobalMem除以1048576.0f转换为MB
        printf("  Total amount of global memory:                 %.0f MBytes " "(%llu bytes)\n", static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),(unsigned long long)deviceProp.totalGlobalMem);


        // 打印GPU最大时钟频率（每个SM的频率）---原始单位是kHz → 转换为MHz和 GHz
        printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f " "GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        // 打印L2缓存(Cache)大小（多用于全局内存数据读写缓存）
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        
        // ★：每个Block可用的Shared Memory总量（程序常用于线程间通信与局部缓存）
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);

        // ★：每个SM（Streaming Multiprocessor）上的共享内存总量
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);

        // ★：每个Block可使用的寄存器数量（影响线程调度效率）
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);

        // ★：Warp大小：通常是 32，表示同时调度的线程组大小
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);

        // ★：每个SM最多能同时容纳的线程数。
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);

        // ★：每个Block中线程数的上限
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

        // ★：每个Block的线程维度限制
        printf("  Max dimension size of a block size(x,y,z):     (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        // ★：整个Grid的网格布局限制
        printf("  Max dimension size of a grid size(x,y,z):      (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    }

    return 0;
}
