/*
=======向量化load & store + 向量加法(基础版) + 测量GPU显存带宽=======

1. float *x和float4 *x的差异是在于：
float *x是以标量形式组织的，那么如果需要存储4个浮点数，则需要从x[0]，x[1]，x[2]，x[2]中逐一创建。
float4 *x则是以向量的形式组织的，包含x，y，z，w，如果需要存储4个浮点数，则只需要从x[0]中创建x[0].x，x[0].y，x[0].z，x[0].w。

个人理解：就像是打包操作，减少了block数量，一次性打包4组数据，然后在内部拆开计算，最后又一次性打包成一个float4 *x。
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ===================== 错误检测 ==================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
// =================================================================



__global__ void vector_add_upgrade(float *gpu_x, float *gpu_y, float *gpu_z, int N_OFFSET)
{
    // 如果在启动核函数时，grid参数没有除4，则线程范围为：[0-10000128)，如果除4后，线程范围为:[0-2499840)，如果向上取整的话，线程范围应该为：[0-2500096)
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;    
    // 情况一和情况二：当grid没有除4(线程范围:[0-10000128]) ||  当grid除4但向上取整时，线程数 > 数据量;(线程范围:[0-2500096)) 
    
    if(global_idx < N_OFFSET / 4)   // 因为线程数 > 数据量，所以1个线程对应1个数据，又因为1个float4*对应4个float*，故只取[0-2500000)个数据和线程即可，后续的线程空闲状态。
    {
        float4 gpu_a = reinterpret_cast<float4 *>(gpu_x)[global_idx];    // 向量化的读取A数组数据，隐式地处理了4个数
        float4 gpu_b = reinterpret_cast<float4 *>(gpu_y)[global_idx];    // 向量化的读取B数组数据，隐式地处理了4个数
        float4 gpu_c;                                                    // 定义C数据存储计算后的数
        gpu_c.x = gpu_a.x + gpu_b.x;
        gpu_c.y = gpu_a.y + gpu_b.y;
        gpu_c.z = gpu_a.z + gpu_b.z;
        gpu_c.w = gpu_a.w + gpu_b.w;
        reinterpret_cast<float4 *>(gpu_z)[global_idx] = gpu_c;           // 向量化的写入
    }
    

    // 情况三：当grid除4且没向上取整时，线程数 < 数据量

    // 这里global_idx的线程范围：[0-2499840),同上，只需要1/4的数据和线程即可。
    // 但是我们发现，缺少了160个线程，因此，我们在后面添加了blockDim.x * gridDim.x(=2499840)，因此0-159号线程要处理两次数据。这就是当线程数<数量时的常用方法。
    // for (int i = global_idx; i < N_OFFSET / 4; i += blockDim.x * gridDim.x)   
    // {       
    //        // 原本10000000个数据应该是每个元素都对应1个线程，现在只需要1/4个线程，因为1个线程现在可以处理4个数据，所以在启动核函数时需要除4。
    //        float4 gpu_a = reinterpret_cast<float4 *>(gpu_x)[i];    // 向量化的读取，隐式地处理了4个数
    //        float4 gpu_b = reinterpret_cast<float4 *>(gpu_y)[i];    // 向量化的读取，隐式地处理了4个数
    //        float4 gpu_c;                                           // 向量化的读取
    //        gpu_c.x = gpu_a.x + gpu_b.x;
    //        gpu_c.y = gpu_a.y + gpu_b.y;
    //        gpu_c.z = gpu_a.z + gpu_b.z;
    //        gpu_c.w = gpu_a.w + gpu_b.w;
    //        reinterpret_cast<float4 *>(gpu_z)[i] = gpu_c;           // 向量化的写入
    //  }
    
}


int main()
{
    
    int N = 100000000;
    int N_OFFSET = 10000000;
    int block_num= 256;
    // 当N和N_OFFSET是整数倍的关系，并且N_OFFSET和256也是整数倍关系，那么可以直接用N_OFFSET/256即可。
    // 初学CUDA编程理解：这里除以4是因为后面核函数中用float4*代替了float*，一个float4*中存在4个float*，因此线程数可以减少为原来的1/4；如果不除4也不会报错，但是，消耗的资源就不一样了。
    
    // 情况一：当不除4时，线程数 > 数据量
    //int grid_num_one = (N_OFFSET + block_num - 1.) / block_num;       
    // 情况二：当除4且向上取整时，线程数 > 数据量
    int grid_num_one = ceil(((N_OFFSET + block_num - 1.) / block_num) / 4);      

    // 情况三：当除4且没向上取整时，线程数 < 数据量
    //int grid_num_one = ((N_OFFSET + block_num - 1.) / block_num) / 4;    

    // 情况三(一半的情况)：当线程只有数量的一半时
    //int grid_num_one = ceil(((N_OFFSET + block_num - 1.) / block_num) / 8);   


    printf("%d\n",grid_num_one);
    dim3 grid_one(grid_num_one);
    float *cpu_x, *cpu_y, *cpu_z;
    float *gpu_x, *gpu_y, *gpu_z;
    float *gpu_result;
    int nbytes = N * sizeof(float);
    cpu_x = (float *)malloc(nbytes);
    cpu_y = (float *)malloc(nbytes);
    cpu_z = (float *)malloc(nbytes);
    gpu_result = (float *)malloc(nbytes);
    cudaMalloc((void**)&gpu_x, nbytes);
    cudaMalloc((void**)&gpu_y, nbytes);
    cudaMalloc((void**)&gpu_z, nbytes);
    for(int i = 0; i < N; i ++)
        {
            cpu_x[i] = 1;
            cpu_y[i] = 2;
            cpu_z[i] = cpu_x[i] + cpu_y[i];
        }
    cudaMemcpy(gpu_x, cpu_x, nbytes, cudaMemcpyHostToDevice);  
    cudaMemcpy(gpu_y, cpu_y, nbytes, cudaMemcpyHostToDevice);
    // printf("warm up start\n");
    vector_add_upgrade<<<grid_one, block_num >>>(gpu_x, gpu_y, gpu_z, N_OFFSET);   // 热身调用，它的作用是为了后续测量纯计算性能，因为首次调用存在初始化的开销，如果不热身，则初始化开销的时间也会计算到性能中。
    // printf("warm up end\n");


    float milliseconds = 0.0; // 它必须设定为float类型,因为后面的cudaEventElapsedTime算出的时间是毫秒级别。

    cudaEvent_t start, stop; 


    cudaEventCreate(&start); 
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    //printf("grid_one:%d\t%d\n",grid_one, N_OFFSET);
    for (int i = N/N_OFFSET - 1; i >= 0; --i) 
    {
        vector_add_upgrade<<<grid_one, block_num >>>(gpu_x + i * N_OFFSET, gpu_y + i * N_OFFSET, gpu_z + i * N_OFFSET, N_OFFSET);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);





    cudaMemcpy(gpu_result, gpu_z, nbytes, cudaMemcpyDeviceToHost);
    
    // 定义一个计数变量，用于计算gpu算出的结果和cpu算出的结果不一致的数量。
    int error_count = 0; 

    // 检查gpu计算得到的每一个结果和cpu计算得到的每一个结果是否一致。
    for(int i = 0; i < N; i++)
    {
        if(fabs(cpu_z[i] - gpu_result[i]) > 1e-6)  // 当差值大于1的-6次方误差就大了，可以判别为不一致。
        {
            printf("Result verification failed at element index %d!\n", i);  // 输出错误的id
            error_count += 1;  // 错误数量累加1
        }
    }

    if(error_count == 0)  // 如果全都正确，则输出结果正确
    {
        printf("Result right\n");
    }

    unsigned N_all = N * 4;
	
	/* 测量显存带宽时, 根据实际读写的数组个数, 指定下行是 1*(float)N 还是 2*(float)N 还是 3*(float)N */
	printf("Mem BW= %f (GB/sec)\n", 3 * (float)N_all / milliseconds / 1e6);


    free(cpu_x);
    free(cpu_y);
    free(cpu_z);
    free(gpu_result);

    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);
    

    return 0;
}
