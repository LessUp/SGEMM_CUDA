#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int K9_NUM_THREADS = 256;

/*
 * ============================================================================
 * Kernel 9：sgemmAutotuned —— 参数化 + 自动调优
 * ============================================================================
 *
 * 作用
 * ----
 * 把前面 kernel 的结构（2D 分块 + 向量化访存 + 转置 A）全部参数化：
 * 所有分块尺寸（BM/BN/BK/TM/TN）都做成模板参数，并用 __launch_bounds__
 * 固定线程数。这样同一个 kernel 源码可以编译出无数种配置，配合
 * scripts/kernel_9_autotuner.sh 在目标 GPU 上穷举搜索最优组合。
 *
 * 为什么需要自动调优
 * ------------------
 * 最优分块参数取决于具体 GPU 的硬件特性（共享内存大小、寄存器数量、
 * SM 数量、带宽等），手工一个个试非常耗时。把参数变成编译期常量后，
 * 脚本可以：枚举参数组合 -> 跳过不满足整除约束的组合 -> sed 改写常量 ->
 * make 重编译 -> 跑基准记录性能。A6000 上搜索出的最优配置是
 * BM=BN=128、BK=16、TM=TN=8（见 runner.cu 中的 K9_* 常量）。
 *
 * 本 kernel 与 kernel 5/6 的关系
 * ------------------------------
 * 结构上仍是"每个线程算 TM×TN 个结果"的 2D 分块：把 C 切成 BM×BN 的
 * 子块，沿 K 维以 BK 为步长滑动，A 转置存入共享内存、B 原样存入。
 * 不同点：
 *   1. 全部尺寸参数化，可自由组合；
 *   2. 引入"线程束分块（warp tiling）"的骨架：把 BM×BN 的子块按
 *      WM=TM×16、WN=TN×16 的逻辑单位切分（WMITER=BN/BM 维上的迭代数），
 *      为 kernel 10 的 warp 级分块做准备——在默认配置下 WMITER=WNITER=1，
 *      等价于没有 warp 级切分；
 *   3. __launch_bounds__(256) 向编译器声明线程块规模，允许它分配更多
 *      寄存器（本 kernel 每线程要同时"活" 2×TM×TN 个 float）。
 * ============================================================================
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 线程束分块（warptile）的尺寸：一个 warp 在 M/N 方向上负责的跨度
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // 一个线程块的分块需要迭代几个 warptile（默认配置下为 1）
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // 本线程在 warptile 中的位置
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // 在共享内存中为当前分块分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 把指针移动到本线程块负责的 A 行、B 列的起点
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 计算本线程负责加载到共享内存的位置索引（float4 向量化加载）
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // 在寄存器文件中为结果分配线程私有的缓存
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 最外层循环：沿 K 维逐个处理分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 把当前分块从全局内存搬运到共享内存（多元素循环，保证合并访问）
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // A 在写入时转置（列优先存储，K 方向连续）
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // B 按原布局整体搬入
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // 按 warptile 迭代计算本线程的结果
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // 把共享内存中的当前列读进寄存器
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          // 寄存器中的 TM×TN 次乘加（FMA）
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    // 指针前进到下一个 K 分块
    A += BK;     // 右移 BK 列
    B += BK * N; // 下移 BK 行
  }

  // 把结果写回全局内存（float4 向量化写回）
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      // 指向当前 warptile 的 C 子块起点
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 把 C 的旧值读进寄存器
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          // 在寄存器中完成 alpha/beta 更新
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // 写回全局内存
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}
