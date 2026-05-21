/*
 计算      算力
—————— < ———————— = 访存密集型算子(elementwize)：从访存的角度优化，可以利用shared memory、向量化load和store
 访存      带宽



 计算      算力
—————— > ———————— = 计算密集型算子(矩阵乘mutmul、卷积conv)：从计算的角度优化，提高并行度，即尽可能提供GPU资源分配尽可能多的block，避免warp divergence，让空闲线程干活
 访存      带宽



 计算      算力
—————— = ———————— = Latency型算子
 访存      带宽

*/


#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cstdlib>
#include <iostream>



constexpr float ALPHA = 0.7978845608028654f;
constexpr float BETA  = 0.044715f;



#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA error at " << __FILE__ << ":"       \
                 << __LINE__ << " - "                         \
                 << cudaGetErrorString(err) << endl;          \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)


    

float gelu_cpu(float x) {

    float tanh_in = ALPHA * (x + BETA * x * x * x);
    return 0.5f * x * (1.0f + tanhf(tanh_in));
}





template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  __host__ __device__ inline const T& operator[](int i) const { 
    return val[i]; 
  }

  __host__ __device__ inline T& operator[](int i) { 
    return val[i]; 
  }
};




__device__ __forceinline__ __half gelu_half_scalar(__half x) {
    float v = __half2float(x);

    constexpr float ALPHA = 0.7978845608028654f;
    constexpr float BETA  = 0.044715f;

    float tanh_in = ALPHA * (v + BETA * v * v * v);
    float out = 0.5f * v * (1.0f + tanhf(tanh_in));

    return __float2half(out);
}


__device__ __forceinline__ void gelu_half2_apply(
    __half* y,
    const __half* x
) {
    half2 x2 = *reinterpret_cast<const half2*>(x);

    float2 xf = __half22float2(x2);

    float y0 = 0.5f * xf.x *
        (1.0f + tanhf(0.7978845608028654f *
        (xf.x + 0.044715f * xf.x * xf.x * xf.x)));

    float y1 = 0.5f * xf.y *
        (1.0f + tanhf(0.7978845608028654f *
        (xf.y + 0.044715f * xf.y * xf.y * xf.y)));

    float2 yf;
    yf.x = y0;
    yf.y = y1;

    half2 y2 = __float22half2_rn(yf);

    *reinterpret_cast<half2*>(y) = y2;
}



// template <int vecSize>
// __global__ void gelu_device(const float* d_x, float* d_y, int N) {

//     /* 
//     // v0：baseline，最朴素的gpu并行思想
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

//     if (global_idx < N) {

//         float x = d_x[global_idx];
//         float tanh_in = ALPHA * (x + BETA * x * x * x);
//         d_y[global_idx] = 0.5f * x * (1.0f + tanhf(tanh_in));
//     }
//     */



//     /*
//     // v1：升级为grid-stride loop，当线程数小于数据量时可以让一个线程处理多个数据
    
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
    
//     for(int i = global_idx; i < N; i += stride){

//         float x = d_x[i];
//         float tanh_in = ALPHA * (x + BETA * x * x * x);
//         d_y[i] = 0.5f * x * (1.0f + tanhf(tanh_in));

//     }
//     */



//     /*
//     // v2： 要让1个线程干多个活，也就是让一个线程一次读取连续的4组数据，和之前向量化读写写法不一样，但是是等价的，例如线程0读取x[0]~x[3],线程1读取x[4]~x[7]
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;  // 0号线程、1号线程……global_idx号线程
//     int offset = global_idx * vecSize;    // 0号线程处理的0号数组下标、1号线程处理的4号数组下标……，之前向量化读写是1个线程处理1个float4，因此只做的是/4，这个读写是1个线程处理4个float，所以做的是*4

//     int stride = blockDim.x * gridDim.x * vecSize;
    
//      for (; offset < N; offset += stride) {

//         for (int i = 0; i < vecSize; ++i) {   // 指代vecSize个数据为1组
//             int idx = offset + i;

//             if (idx < N) {
//                 y[idx] = gelu(x[idx]);
//             }
//         }
//     }
//     */



//     /*
//     // v3：将循环读写升级成向量化的读写(和之前的向量化读写也有点差异)
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int offset = global_idx * VecSize;
//     int stride = blockDim.x * gridDim.x * VecSize;

//     using VecT = AlignedVector<float, VecSize>;

//     for(; offset < N; offset += stride){
//         if(offset + VecSize <= N){
//             VecT x_vec = *reinterpret_cast<const VecT*>(x + offset);  // 当前线程从 x + offset 位置一次读入 VecSize 个 float放到x_vec里面
//             VecT y_vec;

//             #pragma unroll
//             for(int i = 0; i < VecSize; ++i){
//                 y_vec[i] = gelu(x_vec[i]);   // 对 x_vec 里的每个元素做 GELU，结果放到 y_vec
//             }

//             *reinterpret_cast<VecT*>(y + offset) = y_vec;   // 一次性写回 y + offset
//         }else{
//             #pragma unroll
//             for(int i = 0; i < VecSize; ++i){
//                 int idx = offset + i;

//                 if(idx < N){
//                     y[idx] = gelu(x[idx]);
//                 }
//             }
//         }
//     }
//     */
// }





// template<int VecSize>
// __global__ void gelu_half_vec_kernel(const __half* x, __half* y, int N) {

//     /*
//     // v4：将gelu转为half版本---half 输入，转成 float 算 GELU，再转回 half 输出
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int offset = global_idx * VecSize;
//     int stride = blockDim.x * gridDim.x * VecSize;

//     using VecT = AlignedVector<__half, VecSize>;

//     for (; offset < N; offset += stride) {
//         if (offset + VecSize <= N) {
//             VecT x_vec = *reinterpret_cast<const VecT*>(x + offset);
//             VecT y_vec;

//             #pragma unroll
//             for (int i = 0; i < VecSize; ++i) {
//                 y_vec[i] = gelu_half_scalar(x_vec[i]);
//             }

//             *reinterpret_cast<VecT*>(y + offset) = y_vec;
//         } else {
//             #pragma unroll
//             for (int i = 0; i < VecSize; ++i) {
//                 int idx = offset + i;

//                 if (idx < N) {
//                     y[idx] = gelu_half_scalar(x[idx]);
//                 }
//             }
//         }
//     }
//     */


//     /*
//     // V5：利用CUDA的half2 intrinsic将两个half打包成一个half2，一条指令一次处理两个FP16
//     int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int offset = global_idx * VecSize;
//     int stride = blockDim.x * gridDim.x * VecSize;

//     for (; offset < N; offset += stride) {
//         if (offset + VecSize <= N) {
//             #pragma unroll
//             for (int i = 0; i < VecSize; i += 2) {
//                 gelu_half2_apply(y + offset + i, x + offset + i);
//             }
//         } else {
//             #pragma unroll
//             for (int i = 0; i < VecSize; ++i) {
//                 int idx = offset + i;
//                 if (idx < N) {
//                     y[idx] = gelu_half_scalar(x[idx]);
//                 }
//             }
//         }
//     }
//     */

// }  




int main() {

    int N = 1000;

    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *y_gpu = (float*)malloc(N * sizeof(float));


    for (int i = 0; i < N; i++) {
        x[i] = i + 0.1f;
        y[i] = gelu_cpu(x[i]);
    }


    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(float)));


    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));



    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    const int blockSize = std::min<int>(256, deviceProp.maxThreadsPerBlock); 
    dim3 block(blockSize);

    int gridSize = std::min<int>((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    // 向量化读去时只需要vecSize个线程
    // int gridSize = std::min<int>((N + blockSize * vecSize - 1) / (blockSize * vecSize), deviceProp.maxGridSize[0]);
    dim3 grid(gridSize);

    

    gelu_device<<<grid, block>>>(d_x, d_y, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));



    for (int i = 0; i < N; i++) {
        
        float diff = fabsf(y_gpu[i] - y[i]);
        float tol = 1e-5f * fmaxf(1.0f, fabsf(y[i]));

        if (diff > tol) {
            std::cout << "error at " << i
                 << ", cpu = " << y[i]
                 << ", gpu = " << y_gpu[i]
                 << ", diff = " << diff
                 << ", tol = " << tol << std::endl;
            return -1;
        }
    }

    std::cout << "result correct!" << std::endl;

    free(x);
    free(y);
    free(y_gpu);

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
