#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * ============================================================
 * kernel 11：双缓冲（Double Buffering）版 SGEMM
 * ============================================================
 *
 * 【作用】
 * 在 kernel 10（线程束分块）的基础上引入双缓冲（double buffering），
 * 把"全局内存（global memory）→ 共享内存（shared memory）"的加载与
 * "共享内存 → 寄存器的计算"重叠起来，隐藏全局内存的访问延迟。
 * 本文件导出 kernel 函数 sgemmDoubleBuffering。
 *
 * 【解决的核心性能问题】
 * kernel 10 的主循环是"加载 → 同步 → 计算 → 同步 → 加载 → ..."严格串行：
 * 加载下一个 K 分块时，所有线程只能空等全局内存返回数据。全局内存延迟
 * 高达数百个周期，而一个分块的计算只有几十个周期，访存延迟成为整个
 * kernel 的瓶颈。
 *
 * 【优化思路】
 * - 共享内存容量翻倍：As[2*BM*BK]、Bs[2*BK*BN]，两个缓冲区交替使用——
 *   一个正在被计算消费时，另一个可以同时被加载。
 * - 线程分工（软件流水线）：用 doubleBufferIdx 把线程分成两半。前一半
 *   线程先算"当前分块（缓冲 0）"，与此同时后一半线程加载"下一个分块
 *   （缓冲 1）"——同一时刻一半线程在算、一半线程在搬，加载延迟被计算
 *   时间掩盖。两半再各自处理剩下的分块，最后前一半线程把"当前+2"的
 *   分块加载进已被消费完的缓冲 0，完成乒乓（ping-pong）轮转。
 * - 循环步长变为 2*BK：每轮迭代推进两个 K 分块。
 *
 * 【关键概念讲解】
 * - 为什么双缓冲能隐藏延迟？
 *   单缓冲时，计算完当前分块才能加载下一个（否则会覆盖数据），加载完才
 *   能开始计算（否则无数据可用），二者完全串行，总时间 = 加载 + 计算。
 *   双缓冲让二者交叠，总时间 ≈ max(加载, 计算)。这就是流水线（pipeline）
 *   思想：像 CPU 的指令流水线一样，把"取指"和"执行"错开。
 * - 同步点在哪、为什么需要？
 *   凡是"写缓冲"与"读缓冲"交错的地方都要用 __syncthreads 隔开：
 *   ① 前半线程算完缓冲 0、后半线程加载完缓冲 1 之后要同步，确保缓冲 1
 *      已经就绪可供读取；② 所有线程读完缓冲 0 之后要同步，才允许把它
 *      覆盖；③ 循环末尾同步，保证新加载的分块就绪后整块线程再进入下一轮。
 *   注意：同步只保证"数据就绪"，重叠本身靠线程分工实现。
 * - 为什么加载索引要按 NUM_THREADS/2 计算？
 *   加载只由一半线程执行，索引必须按实际干活的人数分配，否则会出现有的
 *   元素没人搬、有的元素被搬两次。
 * - 边界处理：每轮推进两个分块，末尾可能凑不齐，用
 *   if (bkIdx + BK < K) / if (bkIdx + 2*BK < K) 跳过越界的加载与计算。
 * - 本版本的局限：加载仍由线程执行 ld/st 指令，占用线程的指令周期；
 *   下一版（kernel 12）改用异步拷贝（cp.async），把搬运彻底交给硬件。
 * ============================================================
 */

namespace db {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 写入共享内存时顺便完成转置（As 按列优先存放，K 方向连续）
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // B 不需要转置，按原布局整体搬入（float4 一次拷 4 个元素）
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 把当前 K 点（dotIdx）的整个线程束分块取进寄存器：
    // regM 覆盖本线程束的 WM 行，regN 覆盖本线程束的 WN 列
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // 执行本 K 点的线程束分块矩阵乘：regM 的每个元素与 regN 的每个元素
    // 相乘累加，正好凑出本线程负责的全部 TM×TN 结果
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // 逐个结果元素做乘累加
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 本线程所在的线程束在"线程块分块"中的位置
  const uint warpIdx = threadIdx.x / WARPSIZE; // 本线程属于第几个线程束
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // 线程束分块内的子分块尺寸
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 本配置下 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 本配置下 32/2=16

  // 本线程在"线程束子分块"中的位置
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // 如 16/4，取 0~3
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // 如 16/4，取 0~7

  // 双缓冲：共享内存分配两份空间，容量是单缓冲的两倍
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // 用线程 ID 把线程分成两半：前半（0）先算、后半（1）先加载，
  // 让加载与计算在同一时刻重叠
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  // 让 A/B 指针指向本线程块负责的分块：A 的行起点、B 的列起点
  A += cRow * BM * K;
  B += cCol * BN;
  // 让 C 指针指向本线程束负责的输出分块
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算本线程要往共享内存搬运的元素索引。
  // 注意：加载只由一半线程承担，索引按 NUM_THREADS/2 计算
  //（相当于假装线程数只有实际的一半，否则会漏载或重复加载）
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  // 累加结果缓存在寄存器中，避免频繁读写共享内存
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // 按线程束分块粒度缓存的 A/B 片段（寄存器级缓存）
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // 前一半线程先把第 0 个分块（缓冲 0）加载进共享内存
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  }
  __syncthreads(); // 确保第 0 个分块就绪，所有线程才能开始计算

  // 最外层循环：每轮推进两个 K 分块（步长 2*BK）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // 前半线程：先计算当前分块（缓冲 0）
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads(); // 与后半线程的加载汇合：确保缓冲 1 已加载完成

      // 前半线程：再计算下一个分块（缓冲 1）
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
      __syncthreads(); // 所有线程都读完了缓冲 0，才允许覆盖它

      // 前半线程：把"当前+2"的分块加载进缓冲 0，为下一轮做准备
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
            innerRowB, innerColB);
      }
    } else {
      // 后半线程：先加载下一个分块（缓冲 1）——
      // 与前半线程计算缓冲 0 的时间重叠，掩盖加载延迟
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
            innerColA, innerRowB, innerColB);
      }
      __syncthreads(); // 确保缓冲 0 可读、缓冲 1 已加载完成

      // 后半线程：计算当前分块（缓冲 0）
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads(); // 所有线程都读完了缓冲 0，才允许覆盖它

      // 后半线程：再计算下一个分块（缓冲 1）
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    }

    A += 2 * BK;     // 指向右侧的下一个 K 分块
    B += 2 * BK * N; // 指向下方的下一个 K 分块
    __syncthreads(); // 新加载的缓冲就绪后，整块线程再进入下一轮
  }

  // 把寄存器中的结果写回全局内存（C 矩阵）
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 把 C 指针移动到当前线程束子分块的起始位置
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 先把 C 的旧值读进寄存器（float4 向量读）
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // 在寄存器中完成 alpha/beta 更新
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // 写回全局内存
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}
