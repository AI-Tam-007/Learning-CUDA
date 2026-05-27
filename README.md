# **Learning-CUDA**

从零开始系统学习 CUDA 编程的实战项目，从入门 HelloCUDA 到常用算子（向量加法、规约、直方图、GELU、Softmax）逐步进阶，包含基础实现、优化技巧、性能对比，适合 CUDA 初学者和 AI 推理 / 高性能计算方向开发者。

---

## **项目特点**
- ✅ 循序渐进：从最基础的 CUDA 语法到复杂算子优化
- ✅ 实战导向：覆盖深度学习常用算子（GELU、Softmax 等）
- ✅ 完整优化链路：朴素实现 → 共享内存 → Warp/Shuffle → 寄存器优化 → 向量化读写
- ✅ 性能对比：同算子 CPU / 不同 CUDA 优化版本 / PyTorch 原生实现对比
- ✅ 代码简洁易读：注释详细、命名规范、可直接运行

---

## **目录结构**
```text
Learning-CUDA/
├── 01.hello_cuda/                 # CUDA 环境入门、基础语法
├── 02.element_add_one/            # 单元素并行：每个线程处理1个元素
├── 03.vector_add/                # 向量加法：基础并行、内存访问模式优化
├── 04.device_query/               # 查询 GPU 设备信息（算力、显存、SM 数量等）
├── 05.reduce_kernel/              # 规约算法（求和/最大值）：基础→共享内存→Warp Shuffle 优化
├── 06.histogram/                  # 直方图统计：原子操作、共享内存优化
├── 07.gelu/                       # GELU 激活函数：CUDA 实现与优化
├── 08.softmax/                    # Softmax 算子：全链路优化 + 性能对比
└── README.md                      # 项目说明文档


在8192x4096的阶段：
baseline：
1.查看Speed of Light
Compute (SM) Throughput [%]	3.76
Memory Throughput [%]	48.06
结合Roofline分析推测该kernel为Memory Bound
2.查看Memory Workload Analysis
Memory Throughput [Gbyte/second]	60.21 --- 并未达到硬件的理论带宽
Mem Busy [%]	48.06   Mem Pipes Busy [%]	2.99 --- 指令少，访存负担大，通常是非合并访存的问题(因为baseline的思想是一个threads处理一行，在8192x4096，1个线程处理4096列，正好对应Mem Pipes Busy发射指令少，而Mem Busy忙碌)
Sectors/Req = 平均每个内存事务请求，下发多少个32B扇区，当Sectors/Req = 32 说明一次请求要搬运 32 * 32 = 1024 byte 数据，大量无效内存数据被强行读取，有效数据占比极低
3.查看Warp State Statistics
Stall Long Scoreboard占比很重，所以说明每次访问需要等很久，访存模式不理想。
…………………………




Softmax FP32 性能对比：

| Shape       | DType | CPU        | Naive       | Shared      | Warp Shuffle | Register Cache | Vectorized                          | PyTorch     |
|-------------|-------|------------|-------------|-------------|--------------|----------------|-------------------------------------|-------------|
| 128×256     | FP32  | 0.54337 ms | 0.120909 ms | 0.009728 ms | 0.0192512 ms | 0.0078784 ms   | 0.0072704 ms                        | 0.017738 ms |
| 4096×64     | FP32  | 4.87485 ms | 0.0569344 ms| 0.07936 ms  | 0.0102112 ms | 0.0076512 ms   | 0.0075776 ms（一次向量化读取2个float）| 0.078848 ms |
| 1024×1000   | FP32  | 17.83 ms   | 0.662016 ms | 0.0428128 ms| 0.0457632 ms | 0.0290816 ms   | 0.02816 ms                          | 0.030925 ms |
| 1024×1024   | FP32  | 18.2633 ms | 0.893542 ms | 0.0312288 ms| 0.042496 ms  | 0.0277504 ms   | 0.027648 ms                         | 0.032870 ms |
| 8192×4096   | FP32  | 599.802 ms | 6.27671 ms  | 1.07889 ms  | 1.52983 ms   | 0.98537 ms     | 0.962662 ms                         | 1.474842 ms |
