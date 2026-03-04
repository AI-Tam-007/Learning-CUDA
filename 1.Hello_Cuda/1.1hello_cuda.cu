/*
====== 打印Hello CUDA ====== 
总结：
1. 核函数：__global__ void kernel_name(){}
2. 线程索引(一维)：idx = blockDim.x * blockIdx.x + threadIdx.x;
3. blockDim.x：指代一个block中总共有n个线程；  blockIdx.x：指代第n个block；  threadIdx.x：指代第n个block中的第n个线程；
   例如：||||||  ||||||  ||||， 即：blockDim.x = 6； blockIdx.x = 2； threadIdx.x = 4；  故idx = 6 * 2 + 4 = 16；
           0       1      2
4. 启动核函数：kernel_name<<<grid_dim, block_dim>>(args);
5. 显式同步操作(让CPU等GPU执行完)：cudaDeviceSynchronize();

*/
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_cuda()    // 核函数
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;   // 计算每个线程的全局唯一索引

    printf("Hello CUDA!\n");     // 打印Hello CUDA

}

int main()
{
    hello_cuda<<<1,1>>>();    // 启动核函数，此处只有1个block，但至少是存在32个线程的，因为1个block最少存在1个warp，1个warp通常由32个线程组成。所以，此处只使用了1个线程，其他的线程并没有使用。
    cudaDeviceSynchronize();  // 显示同步操作，CPU等待GPU执行完毕

    return 0;
}
