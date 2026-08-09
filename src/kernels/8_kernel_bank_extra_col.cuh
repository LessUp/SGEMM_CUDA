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
 * Kernel 8：sgemmResolveBankExtraCol —— 消除存储体冲突（加列偏移/填充）
 * ============================================================================
 *
 * 作用
 * ----
 * 与 kernel 7 目标相同（消除读取 Bs 时的存储体冲突），但用另一种更简单
 * 的思路：在共享内存数组的每一行末尾额外填充（padding）几个 float，
 * 改变行的步长，让"行号"不再与存储体编号对齐。
 *
 * 原理
 * ----
 * 存储体冲突的本质是"不同行的相同列"落进了同一组 bank：
 *   朴素布局  bank(r, c) = (r * 行宽 + c) % 32
 * 行宽是 32 的整数倍时，(r*32+c)%32 ≡ c%32——同一列的所有元素永远
 * 挤在同一个 bank 上，warp 一读就是一个大冲突（用 scripts/bank_calc.py
 * 的简化模型可以直观看到：16 路冲突，只有 2/32 个 bank 被用到）。
 *
 * 本 kernel 给 Bs 每行多留 extraCols=5 个 float（行宽从 128 变成 133），
 * 由于 133 % 32 = 5 ≠ 0，每换一行 bank 编号就整体偏移 5，行与行之间
 * 不再对齐到同一组 bank，冲突从多路降到 2 路（scripts/bank_calc.py
 * 对比了"行宽 32"与"行宽 33"两种布局的冲突数）。
 *
 * 与 kernel 7 的对比
 * ------------------
 *   - kernel 7（线性化）：不浪费共享内存，但写入索引复杂、指令多；
 *   - kernel 8（加列偏移）：索引简单直观，代价是多占约 5~10% 的共享
 *     内存（128 行宽每行多 5 个 float）。实测两者性能几乎相同，
 *     本 kernel 略高（见 README 基准表）。
 *
 * 代码要点
 * --------
 * - Bs 的声明与读写全部使用 (BN + extraCols) 作为行宽；
 * - 共享内存大小：BK * (BN + extraCols)，比实际需要多出 BK*5 个 float；
 * - As 不填充（它的读取模式本身无冲突），保持原样。
 * ============================================================================
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN 是每一列需要占用的线程数
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 在共享内存中为当前分块分配空间。
  // Bs 每行多留 extraCols 个 float（这里是 5），行宽变为 BN+5
  __shared__ float As[BM * BK];
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

  // 把指针移动到本线程块负责的 A 行、B 列的起点
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 计算本线程负责加载到共享内存的位置索引（float4 向量化加载）
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

    // B 按"行宽 = BN+5"的布局原样搬入（float4 一次拷 4 个元素）
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
    __syncthreads();

    // 指针前进到下一个 K 分块
    A += BK;     // 右移 BK 列
    B += BK * N; // 下移 BK 行

    // 计算本线程的结果（读取 Bs 时同样使用 padded 行宽）
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 把共享内存中的当前列读进寄存器
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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
