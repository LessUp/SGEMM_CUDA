#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * Kernel 2：sgemm_global_mem_coalesce —— 全局内存合并访问
 * ============================================================================
 *
 * 作用
 * ----
 * 仍然每个线程算 C 的一个元素，但改变线程到元素的映射，让同一个线程束
 * （warp，32 个连续 threadIdx）读取的地址尽量连续，满足全局内存的
 * 合并访问（coalescing）要求。
 *
 * 解决的核心性能问题
 * ------------------
 * 朴素实现（kernel 1）中，线程束内相邻线程的 x（行）递增、y（列）相同：
 * 访问 A 的 A[x*K + i] 时，x 每 +1 地址就跳过一整行（间隔 K 个 float），
 * 一个 warp 的 32 次访问散布在 32×K×4 字节的范围内，硬件要为每个
 * 32 字节段发一次访存请求，有效带宽只有理论值的零头；写回
 * C[x*N + y] 同样是步长 N 的分散写。访问 B 的 B[i*N + y] 时 y 相同，
 * 32 个线程读的是同一地址（广播），一次事务虽能服务整个线程束，
 * 但每次只带回 4 字节的有用数据。
 *
 * 本 kernel 的做法：让线程束内线程的列索引 cCol 连续（同一行内、
 * cRow 相同），读 B[i*N + cCol] 时地址连续，一次访存请求即可取回
 * 整段数据；读 A[cRow*K + i] 时 cRow 相同 → 广播，一次事务服务
 * 整个线程束。
 *
 * 线程到元素的映射
 * ----------------
 * 一个线程块负责 BLOCKSIZE×BLOCKSIZE 的输出子块，块内 BLOCKSIZE² 个
 * 线程（runner 中以 BLOCKSIZE=32 调用，即 1024 个线程；threadIdx.x
 * 是线性的，靠除法/取模解出二维坐标）：
 *   cRow = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE
 *   cCol = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE
 * 注意 threadIdx.x 连续递增时，先走完一行（cRow 固定、cCol 从 0 到
 * BLOCKSIZE-1），再换下一行——这就是"线程编号从小到大对应地址从
 * 小到大"的合并访问。（blockIdx.x 负责行方向、blockIdx.y 负责列方向，
 * 与 kernel 1 的网格划分一致。）
 *
 * 为什么还不够快
 * --------------
 * 合并访问只解决了"访存效率"，没有解决"访存量"：每个线程每轮点积迭代
 * 仍要发 2 次全局内存读取（一次 A、一次 B），读取总量一分没少；而且
 * A、B 元素被块内多个线程重复读取，只能靠 L1/L2 缓存兜底。kernel 3
 * 开始用共享内存分块来降低全局内存的读取总量。
 * ============================================================================
 */

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // 线程到元素的映射：threadIdx.x 连续递增时先铺满一行（cCol 从 0 到
  // BLOCKSIZE-1），再进入下一行（cRow 递增），保证相邻线程访问连续地址
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // 越界检查：矩阵尺寸不是线程块尺寸整数倍时拦下多余线程
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    // 点积：相邻线程读 B 的地址连续（cCol 连续）、读 A 的地址相同
    // （cRow 相同，硬件按广播处理）
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
