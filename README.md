# 从零手写高性能 CUDA SGEMM

用 CUDA 一步步优化矩阵乘法（SGEMM，单精度通用矩阵乘）的实战教程项目。

每个 kernel 的详细原理讲解见 [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)（英文博客，
建议配合本项目代码阅读）。

## 性能概览

在 NVIDIA A6000（Ampere 架构）上的实测结果：

![](benchmark_results.png)

矩阵尺寸 4096x4096 时的性能（GFLOPs）：

<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | 相对 cuBLAS 的性能 |
|:------------------------------------|----------:|:-------------------|
| 1: 朴素实现 (Naive)                 |   `309.0` | 1.3%               |
| 2: 全局内存合并访问 (GMEM Coalescing) |  `1986.5` | 8.5%               |
| 3: 共享内存缓存 (SMEM Caching)      |  `2980.3` | 12.8%              |
| 4: 一维分块 (1D Blocktiling)        |  `8474.7` | 36.5%              |
| 5: 二维分块 (2D Blocktiling)        | `15971.7` | 68.7%              |
| 7: 避免存储体冲突-线性化索引        | `16213.4` | 69.7%              |
| 8: 避免存储体冲突-加列偏移          | `16459.2` | 70.8%              |
| 11: 双缓冲 (Double Buffering)       | `17278.3` | 74.3%              |
| 6: 向量化访存 (Vectorized Mem Access) | `18237.3` | 78.4%              |
| 9: 自动调优 (Autotuning)            | `19721.0` | 84.8%              |
| 10: 线程束分块 (Warptiling)         | `21779.3` | 93.7%              |
| 0: cuBLAS（NVIDIA 官方库）          | `23249.6` | 100.0%             |
<!-- benchmark_results -->

> 上表是 `gen_benchmark_results.sh` 自动生成的，改代码后重新跑基准即可刷新。
> 注意 6 号 kernel（向量化）没有写在最前，因为它的优化依赖前面几个 kernel 打下的基础，
> 实际教学顺序是：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11。

### 学习路线：12 个 kernel 讲了什么

| 编号 | 文件 | 优化主题 | 一句话总结 |
|:----:|------|---------|-----------|
| 1 | `1_naive.cuh` | 基线 | 最直观的实现，每个线程算一个输出元素，性能很差 |
| 2 | `2_kernel_global_mem_coalesce.cuh` | 全局内存合并访问 | 让一个 warp 的访存落在连续的地址上 |
| 3 | `3_kernel_shared_mem_blocking.cuh` | 共享内存分块 | 把 A、B 的子块先搬到共享内存，减少全局内存访问 |
| 4 | `4_kernel_1D_blocktiling.cuh` | 一维分块 | 每个线程用寄存器数组算一行 C，减少共享内存读次数 |
| 5 | `5_kernel_2D_blocktiling.cuh` | 二维分块 | 每个线程算 TM×TN 的 C 子块，数据复用更充分 |
| 6 | `6_kernel_vectorize.cuh` | 向量化访存 | 用 float4 一次搬 4 个元素，减少访存指令数 |
| 7 | `7_kernel_resolve_bank_conflicts.cuh` | 消除存储体冲突（线性化） | 重新排列共享内存布局，避免 warp 访问同一 bank |
| 8 | `8_kernel_bank_extra_col.cuh` | 消除存储体冲突（加偏移） | 共享内存数组每行多留一列，天然错开 bank |
| 9 | `9_kernel_autotuned.cuh` | 自动调优 | 把分块尺寸做成模板参数，脚本穷举找最优配置 |
| 10 | `10_kernel_warptiling.cuh` | 线程束分块 | 每个 warp 独立算一个大子块，减少同步开销 |
| 11 | `11_kernel_double_buffering.cuh` | 双缓冲 | 算当前分块的同时预取下一个分块，隐藏加载延迟 |
| 12 | `12_kernel_double_buffering.cuh` | 双缓冲进阶 | 11 的变体，两版在同步方式上不同，可对比学习 |

一条性能主线贯穿始终：**限制 SGEMM 性能的从来不是算力，而是数据搬运**。
从 309 GFLOPs 到 21779 GFLOPs（70 倍），每一步都在让数据更快地流向计算单元：
全局内存 → 共享内存 → 寄存器，每一层都比上一层快几个数量级，但容量也更小。

## 环境准备

1. 安装依赖：CUDA toolkit 12、Python（含 Seaborn）、CMake、Ninja。参考 [environment.yml](environment.yml)。
1. 配置 NVCC 编译参数。先查你的 GPU 的计算能力（compute capability）
   [在这里查询](https://developer.nvidia.com/cuda-gpus)，然后打开 `CMakeLists.txt` 修改：
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
    > 计算能力要和你的 GPU 匹配，比如 A6000/RTX 3090 是 80（Ampere），RTX 4090 是 89（Ada）。
1. 编译：`mkdir build && cd build && cmake .. && cmake --build .`
1. 运行某个 kernel：`DEVICE=<device_id> ./sgemm <kernel 编号>`
   （编号 0-12，0 是 cuBLAS 基线，1-12 对应上表）
1. 用 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)（ncu）做性能分析：
   `make profile KERNEL=<kernel 编号>`

基准测试框架来源于 [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)，在此致谢。
