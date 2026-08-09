#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * Kernel 5：sgemm2DBlocktiling —— 二维分块（2D blocktiling）
 * ============================================================================
 *
 * 作用
 * ----
 * 在 kernel 4（一维分块）的基础上，把"每个线程算一行中 TM 个结果"
 * 升级为"每个线程算一个 TM×TN 的二维结果子块"。线程块内的线程按
 * (BM/TM)×(BN/TN) 的网格排列，共同覆盖整个 BM×BN 的 C 分块。
 *
 * 解决的核心性能问题
 * ------------------
 * kernel 4 里每个结果平均要从共享内存读约 BK 个数（A 值用一次就丢，
 * 只有 B 值被复用）。而乘加（FMA）指令本身是流水线化的、几乎免费，
 * 真正的瓶颈是喂数据的带宽——共享内存读得越少，算得越快。
 *
 * 本 kernel 让每个从共享内存读出的数在寄存器里被反复使用：
 *   - A 的一个值 regM[i] 依次与 TN 个 B 值相乘，参与 TN 个结果；
 *   - B 的一个值 regN[j] 依次与 TM 个 A 值相乘，参与 TM 个结果。
 * 于是每个结果平均只需 (BK/TN + BK/TM) 次共享内存读取——以 TM=TN=8、
 * BK=16 为例，从 kernel 4 的约 18 次/结果降到 4 次/结果。数据复用从
 * "共享内存层面"深入到了"寄存器层面"。
 *
 * 相对上一个 kernel 的改进
 * ------------------------
 * 1. 二维结果分块（TM×TN）：核心收益是提高"计算强度"——同样多的
 *    内存流量产出成倍的乘加运算。每线程的结果数越多，数据复用率越高；
 * 2. 显式寄存器缓存 regM[TM]、regN[TN]：每个点积迭代先从共享内存
 *    各读一次，之后 TM×TN 个乘加全部在寄存器之间完成；
 * 3. 共享内存的加载改为多元素循环：每个线程沿行方向加载多个元素
 *    （strideA/strideB 控制每次覆盖的行数），保证每一次加载都覆盖
 *    完整的行宽、地址连续，继续满足合并访问（coalescing）；
 * 4. __launch_bounds__((BM*BN)/(TM*TN), 1)：向编译器声明本 kernel 的
 *    线程块规模上限和每 SM 期望的最小块数，让编译器敢于分配更多
 *    寄存器（本 kernel 每线程同时要"活" 2×TM×TN+TM+TN 个 float）。
 *
 * 关键概念：为什么"每线程算得越多越快"？
 * --------------------------------------
 * SGEMM 的优化本质是"用片上存储换全局内存带宽"。寄存器是零延迟的，
 * 把 A、B 的值读进寄存器后，一次读取可以支撑多个乘加。TM×TN 越大，
 * 每个寄存器的复用次数越多，内存带宽压力越小，越接近理论峰值。
 *
 * 但注意：寄存器是有限资源（每线程约 255 个）。TM、TN 过大会挤爆
 * 寄存器，导致占用率（occupancy）下降甚至编译失败，所以模板参数
 * 需要在复用率与占用率之间权衡。
 *
 * 存储体冲突提示
 * --------------
 * 读 As 时，相邻线程（threadRow 相差 1）的地址步长是 TM×BK 个 float，
 * 恰为 32（存储体数）的整数倍，会落进同一个存储体，产生存储体冲突
 * （bank conflict，路数取决于模板参数）；读 Bs 时步长为 TN 个 float，
 * 落在不同存储体，无冲突。As 的这个问题会在 kernel 6 中通过"转置
 * 存储"解决。
 * ============================================================================
 */

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // 每个线程负责计算分块内的 TM×TN 个元素
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // 分块结果数 / 每线程结果数 == 线程块线程数
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN 是每一列需要占用的线程数
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 在共享内存中为当前分块分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 把指针移动到本线程块负责的 A 行、B 列的起点
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 计算本线程负责加载到共享内存的位置索引
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // 单个线程块一次加载时，As 中每次跨越的行数（步长）
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // 加载 As 和 Bs 时，希望每次加载都覆盖完整的列宽，以获得更好的
  // 全局内存合并访问（而不是覆盖整行宽度后再逐列迭代）
  const uint strideB = numThreadsBlocktile / BN;

  // 在寄存器文件中为结果分配线程私有的缓存
  float threadResults[TM * TN] = {0.0};
  // As 和 Bs 的寄存器缓存：每个点积迭代把当前列读进寄存器
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 最外层循环：沿 K 维逐个处理分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 把当前分块从全局内存搬运到共享内存
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // 指针前进到下一个 K 分块
    A += BK;     // 右移 BK 列
    B += BK * N; // 下移 BK 行

    // 计算本线程的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 把共享内存中的当前列读进寄存器
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // 把结果写回全局内存
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
