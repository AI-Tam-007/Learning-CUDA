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

---

## **NVIDIA Nsight Compute性能分析**
在8192x4096的阶段：

**Baseline版本：**

1. 查看Speed of Light  
   <img width="1483" height="427" alt="image" src="https://github.com/user-attachments/assets/1603481c-8027-4670-878a-42b101c9c4c1" />  
   Compute (SM) Throughput [%]	3.76  
   Memory Throughput [%]	48.06  
   由ncu数据可知，整体memory 系统压力比 compute 高很多，但是 DRAM 只有 17.48%，说明不是外部显存带宽被打满。  
   <img width="1463" height="367" alt="image" src="https://github.com/user-attachments/assets/4f1c82fd-eb0e-4b1f-be10-d56e3846244b" />  
   结合Roofline分析推测该kernel为Memory Bound。

2. 查看Memory Workload Analysis  
   <img width="1473" height="152" alt="image" src="https://github.com/user-attachments/assets/c10f9005-6acc-4b36-b5db-393b460111b4" />  
   Memory Throughput [Gbyte/second]	60.21 --- 并未达到硬件的理论带宽  
   Mem Busy [%]	48.06   Mem Pipes Busy [%]	2.99 --- 指令少，访存负担大，通常是非合并访存的问题(因为baseline的思想是一个threads处理一行，在8192x4096，1个线程处理4096列，正好对应Mem Pipes Busy发射指令少，而Mem Busy忙碌)  
   Sectors/Req = 平均每个内存事务请求，下发多少个32B扇区，当Sectors/Req = 32 说明一次请求要搬运 32 * 32 = 1024 byte 数据，大量无效内存数据被强行读取，有效数据占比极低  
   查看memory chart：  
   <img width="1124" height="623" alt="image" src="https://github.com/user-attachments/assets/52651ef3-3c6f-46c2-94a6-b13b4243910b" />  
   所有内存操作，都是通过 Global 全局内存指令完成的，其他通道完全没用到。

3. Launch Statistics  
   <img width="1481" height="171" alt="image" src="https://github.com/user-attachments/assets/688ffb97-ce22-49fd-9acf-5851ef4f7716" />  
   Registers Per Thread [register/thread]	39 --- 寄存器处于正常范围内，并不是导致瓶颈的问题  
   Waves Per SM	0.13 --- 这表明SM还没开始发力就结束了，再看到Grid Size	32，这表明grid太小了

4. Occupancy  
   <img width="1479" height="170" alt="image" src="https://github.com/user-attachments/assets/f1b2b939-0a5f-4398-9aef-7c675773fb82" />  
   理论Occupancy达到了100%，但和实际Occupancy对比，差距非常大，说明任务不足或者是调度不够  
   Achieved Active Warps Per SM [warp]	7.99 说明一个SM同时只跑了近8个warp，一般是48个，说明是任务不足  
   从block limit最小的值可以看出，registers和warp是瓶颈。

5. 查看Scheduler Statistics  
   <img width="1468" height="153" alt="image" src="https://github.com/user-attachments/assets/d44d2275-4430-40f9-a47a-12d6d7137065" />  
   No Eligible很高，说明没有准备好的warp，Active Warps Per Scheduler低，说明warp数量不足，Eligible Warps Per Scheduler, Issued Warp Per Scheduler 进一步佐证了结果。

6. 查看Warp State Statistics  
   <img width="1486" height="631" alt="image" src="https://github.com/user-attachments/assets/328014ea-ac77-41ec-9ee8-0a5a73f7312a" />  
   Stall Long Scoreboard占比很重，所以说明每次访问需要等很久，访存模式不理想。

---

## **结果数据对比**

Softmax FP32 性能对比：

| Shape       | DType | CPU        | Naive       | Shared      | Warp Shuffle | Register Cache | Vectorized                          | PyTorch     |
|-------------|-------|------------|-------------|-------------|--------------|----------------|-------------------------------------|-------------|
| 128×256     | FP32  | 0.54337 ms | 0.120909 ms | 0.009728 ms | 0.0192512 ms | 0.0078784 ms   | 0.0072704 ms                        | 0.017738 ms |
| 4096×64     | FP32  | 4.87485 ms | 0.0569344 ms| 0.07936 ms  | 0.0102112 ms | 0.0076512 ms   | 0.0075776 ms（一次向量化读取2个float）| 0.078848 ms |
| 1024×1000   | FP32  | 17.83 ms   | 0.662016 ms | 0.0428128 ms| 0.0457632 ms | 0.0290816 ms   | 0.02816 ms                          | 0.030925 ms |
| 1024×1024   | FP32  | 18.2633 ms | 0.893542 ms | 0.0312288 ms| 0.042496 ms  | 0.0277504 ms   | 0.027648 ms                         | 0.032870 ms |
| 8192×4096   | FP32  | 599.802 ms | 6.27671 ms  | 1.07889 ms  | 1.52983 ms   | 0.98537 ms     | 0.962662 ms                         | 1.474842 ms |
