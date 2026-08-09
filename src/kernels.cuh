// kernels.cuh —— 汇总包含全部 12 个 kernel 的实现头文件
// 每个 kernel 是一个独立的 .cuh 文件，按优化递进顺序编号：
//   1 naive（朴素基线） → 2 合并访问 → 3 共享内存 → 4/5 分块 →
//   6 向量化 → 7/8 消除存储体冲突 → 9 自动调优 → 10 线程束分块 → 11/12 双缓冲
#pragma once

#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"
#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_resolve_bank_conflicts.cuh"
#include "kernels/8_kernel_bank_extra_col.cuh"
#include "kernels/9_kernel_autotuned.cuh"
