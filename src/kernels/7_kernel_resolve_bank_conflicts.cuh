#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * ============================================================================
 * Kernel 7：sgemmResolveBankConflicts —— 消除存储体冲突（线性化布局）
 * ============================================================================
 *
 * 作用
 * ----
 * 在 kernel 6（向量化访存）的基础上，只改共享内存里 B 分块（Bs）的存放
 * 方式，消除读取 Bs 时的存储体冲突（bank conflict）。A 分块（As）的
 * 读取模式没有冲突，保持不变。
 *
 * 背景：什么是存储体冲突？
 * ------------------------
 * GPU 的共享内存在硬件上被分成 32 个存储体（bank），每个 bank 每周期
 * 只能为一个线程返回数据。一次 warp 指令中，若多个线程访问了"不同地址
 * 但同一 bank"，硬件只能把这些访问拆成多轮串行执行——这一轮访问被放大了
 * 几倍，共享内存带宽随之损失。这正是 kernel 6 的问题：
 *   regN[i] = Bs[dotIdx * BN + threadCol * TN + i]
 * 同一 warp 内相邻线程的 threadCol 相差 1，地址相差 TN（本配置 8）个
 * float。32 个线程的地址步长 8，只落在 4 个不同的 bank 上（8×0..7 的
 * bank 编号是 0,8,16,24 循环），形成 8 路冲突——共享内存带宽被吞掉 8 倍。
 *
 * 本 kernel 的解法：线性化（linearize）Bs 的存储布局
 * -------------------------------------------------
 * 不再按"行优先"存 Bs[row * BN + col]，而是重新排布，使得读取时
 * 连续线程访问连续地址（步长 1），彻底打散 bank 分布：
 *   - 写入：Bs[((innerColB % 2) * 4 + innerRowB * 8 + j) * 16 + innerColB / 2]
 *   - 读取：regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol]
 * 读取时 threadCol 是地址里变化最快的下标（步长 1）：一个 warp 的 32 个
 * 线程（threadCol = threadIdx.x % 16，取值 0~15）访问 16 个连续地址，
 * 即 16 个连续 bank——冲突从 8 路降到 2 路（每个地址被两个线程命中，
 * 因为块内列方向的线程数 16 小于 warp 宽度 32）。
 *
 * 代价
 * ----
 * 写入时的索引计算变复杂（多了取模、移位、乘 16），指令数增加；
 * 但写入只发生一次（搬运进共享内存），而读取在点积循环里发生 BK 次，
 * 所以"读得快"的收益远大于"写得多"的代价。相比 kernel 8 的"加列
 * 偏移"方案，本方案不浪费任何共享内存空间。
 * ============================================================================
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN 是每一列需要占用的线程数
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 在共享内存中为当前分块分配空间（Bs 仍按原大小分配）
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 把指针移动到本线程块负责的 A 行、B 列的起点
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 计算本线程负责加载到共享内存的位置索引
  //（float4 向量化：每线程每步搬运 128bit / 32bit = 4 个元素）
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // 在寄存器文件中为结果分配线程私有的缓存
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 最外层循环：沿 K 维逐个处理分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 把当前分块从全局内存搬运到共享内存
    // A 在搬运时转置存储（列优先），保证读取时 K 方向连续
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // 写入 B 时做"线性化"重排：把元素按读取时需要的顺序重新散开，
    // 使后续读取 regN[i] 时 threadCol 下标连续（见上方讲解）
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // 指针前进到下一个 K 分块
    A += BK;     // 右移 BK 列
    B += BK * N; // 下移 BK 行

    // 计算本线程的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 把共享内存中的当前列读进寄存器
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      // 读取 Bs：地址的末位下标是 threadCol（步长 1），
      // warp 内 32 个线程落在 16 个连续 bank 上，冲突仅为 2 路
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
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

  // 把结果写回全局内存（float4 向量化写回）
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 把 C 的旧值读进寄存器
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // 在寄存器中完成 alpha/beta 更新
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // 写回全局内存
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}
