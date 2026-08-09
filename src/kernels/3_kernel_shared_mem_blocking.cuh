#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * ============================================================================
 * Kernel 3：sgemm_shared_mem_block —— 共享内存分块
 * ============================================================================
 *
 * 作用
 * ----
 * 引入共享内存（shared memory）分块：每个线程块负责 C 中一个
 * BLOCKSIZE×BLOCKSIZE 的子块，先把计算所需的 A、B 子块从全局内存
 * 搬进共享内存，再从共享内存读数据做点积。全局内存只在"搬运"时
 * 被访问一次。
 *
 * 解决的核心性能问题
 * ------------------
 * kernel 2 虽然访存连续了，但每个 A/B 元素仍要从全局内存读多次：
 * 一个 A 元素会被同一行的 BLOCKSIZE 个输出元素使用，一个 B 元素会被
 * 同一列的 BLOCKSIZE 个输出元素使用，全部各读一遍。全局内存延迟
 * 数百个周期，反复访问它就是浪费算力。
 *
 * 本 kernel 的思路：数据复用从"反复读全局内存"变成"一次搬运 +
 * 多次命中共享内存"。共享内存是片上存储，延迟约 20~30 个周期、
 * 带宽比全局内存高一个数量级。每个 A 元素只从全局内存读 1 次，
 * 然后被 BLOCKSIZE 个线程共享（读 BLOCKSIZE 次）；
 * 全局内存的读取总量降为原来的约 1/BLOCKSIZE。
 *
 * 访问流程（外层循环沿 K 维以 BLOCKSIZE 为步长滑动）
 * --------------------------------------------------
 *   1. 每个线程把 A、B 各一个元素从全局内存搬进共享内存
 *      （As/Bs，本块内所有线程共享）；
 *   2. __syncthreads()：等整块数据搬完，避免有的线程读到半成品；
 *   3. 每个线程从共享内存读本块数据，做一次长度为 BLOCKSIZE 的点积；
 *   4. __syncthreads()：等所有线程算完，避免快的线程提前覆盖
 *      共享内存里的下一块数据；
 *   5. 指针前进 BLOCKSIZE，处理下一个 K 分块，直到 K 维走完。
 *
 * 合并访问的保证
 * --------------
 * 搬运时让连续线程对应连续地址（threadCol = threadIdx.x % BLOCKSIZE
 * 作为共享内存/全局内存的列下标），一个 warp 的访问落在连续区间，
 * 继续满足合并访问。
 *
 * 局限
 * ----
 * 每个线程每轮只算一个输出元素，点积时对共享内存的访问量仍然很大；
 * kernel 4 用"每线程算一行中的 TM 个结果"来摊薄共享内存读取。
 * ============================================================================
 */

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // 本线程块负责的 C 子块坐标（blockIdx 直接定位子块）
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // 为当前子块在共享内存中分配缓冲区（块内所有线程共享）
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // 本线程在子块内的行列位置
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // 把指针移到本线程块负责的数据起点：
  A += cRow * BLOCKSIZE * K;                    // A 的第 cRow 块行
  B += cCol * BLOCKSIZE;                        // B 的第 cCol 块列
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // C 的 (cRow, cCol) 子块

  float tmp = 0.0;
  // 外层循环：沿 K 维以 BLOCKSIZE 为步长，逐块累加
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // 每个线程搬运一个元素进共享内存。让 threadCol（= threadIdx.x）
    // 作为连续下标，保证全局内存访问的合并（coalescing）
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // 等整块数据搬完才能开始计算（否则可能读到半成品）
    __syncthreads();
    A += BLOCKSIZE;     // A 右移 BLOCKSIZE 列
    B += BLOCKSIZE * N; // B 下移 BLOCKSIZE 行

    // 在共享内存中的当前分块上做点积（长度为 BLOCKSIZE）
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // 再同步一次：防止快的线程提前把下一块数据搬进共享内存，
    // 覆盖掉慢线程还在读的当前块
    __syncthreads();
  }
  // 写回结果（顺带完成 α·(A@B) + β·C）
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
