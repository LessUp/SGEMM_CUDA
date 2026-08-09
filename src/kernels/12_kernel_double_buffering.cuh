#pragma once

#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * ============================================================================
 * Kernel 12：sgemmDoubleBuffering2 —— 双缓冲进阶：cp.async 异步拷贝
 * ============================================================================
 *
 * 【作用】
 * kernel 11 用"线程分工"实现了双缓冲（一半线程算、一半线程搬），但搬运
 * 仍由线程执行普通的 ld/st 指令，占用线程的指令发射周期。本 kernel 把
 * 搬运换成 cuda::memcpy_async（底层是 Ampere 的 cp.async 指令）：拷贝由
 * 硬件在后台完成，发起线程登记一个"完成事件"到 cuda::barrier 上即可
 * 继续做别的事，访存延迟被彻底隐藏。
 *
 * 【解决的核心性能问题】
 * kernel 11 的加载是"用线程的指令周期去等全局内存"：每条加载指令都要
 * 等数百个周期的数据返回，期间线程被占住。cp.async 把"发起拷贝"和
 * "拷贝完成"解耦——发起后线程立刻返回，等真正要用数据时再通过 barrier
 * 一次性等待。线程的算力不再被访存等待浪费。
 *
 * 【与 kernel 11 的对比】
 *   1. 同步机制：kernel 11 用 __syncthreads()（全线程屏障）；本 kernel
 *      用 cuda::barrier 的 arrive_and_wait()，可同时等待"所有线程到达"
 *      与"所有异步拷贝完成"两件事；
 *   2. 搬运方式：kernel 11 是同步 ld/st；本 kernel 是 cp.async（异步，
 *      不占线程）；
 *   3. 线程分工：kernel 11 把线程拆成两半轮换加载/计算；本 kernel 所有
 *      线程同时发起拷贝、同时计算，靠 front/back 两个 barrier 轮流
 *      "保护"两块缓冲区（乒乓）。
 *
 * 【关键概念讲解】
 * - cuda::barrier 是什么？
 *   一种线程块内的同步原语，比 __syncthreads() 更灵活。初始化时设置
 *   "期望到达数"（这里 = block.size()，即所有线程都参与）。每个线程
 *   调用 arrive_and_wait()：arrive 表示"我到达"，wait 表示"等所有线程
 *   都到达、且登记到本 barrier 上的所有异步拷贝都完成"才继续。
 *   注意：barrier 是"分阶段"的（phase），一个阶段结束后自动进入下一
 *   阶段，可以循环复用——这正是双缓冲需要的。
 * - cuda::memcpy_async(..., barrier) 做了什么？
 *   发起一次异步拷贝（全局内存 -> 共享内存），并把这次拷贝的完成
 *   登记到 barrier 上。线程不等拷贝完成就返回，继续执行后续指令。
 * - front/back 两个 barrier 轮流切换（乒乓）：
 *   每轮迭代，frontBarrier 守护"正在被计算"的缓冲区（确保它的拷贝
 *   已完成），backBarrier 守护"正在被加载"的缓冲区（下一轮要用的）。
 *   迭代末尾交换两者的角色（swap 指针），实现缓冲区与屏障的交替复用。
 * - 本版本的局限：
 *   memcpy_async 的完成仍需线程在计算前统一等待（arrive_and_wait），
 *   只是等待变得更"便宜"；完全无等待的流水线需要更精细的软件流水线
 *   编排（如多级 prefetch），超出本教程范围。
 * ============================================================================
 */

namespace {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {

  // 用 memcpy_async 逐元素发起异步拷贝：A 转置写入共享内存（列优先）。
  // 每个拷贝都登记到 barrier，全部完成前线程不会被阻塞
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    cuda::memcpy_async(&As[(innerColA * 4 + 0) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 1) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 1],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 2) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 2],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 3) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 3],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
  }

  // B 不需要转置，按原布局整体搬入（float4 一次拷 4 个元素）
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cuda::memcpy_async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                       &B[(innerRowB + offset) * N + innerColB * 4],
                       cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                       barrier);
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

} // namespace

/*
 * @tparam BM 线程块在 M 维上缓存进共享内存的分块大小
 * @tparam BN 线程块在 N 维上缓存进共享内存的分块大小
 * @tparam BK 线程块在 K 维上缓存进共享内存的分块大小
 * @tparam WM 每个线程束负责的 M 维连续分块大小
 * @tparam WN 每个线程束负责的 N 维连续分块大小
 * @tparam WMITER M 维上子分块的步数（每个线程束要算几个行子块）
 * @tparam WNITER N 维上子分块的步数（每个线程束要算几个列子块）
 * @tparam TM 每个线程在 M 维上的结果分块大小
 * @tparam TN 每个线程在 N 维上的结果分块大小
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  auto block = cooperative_groups::this_thread_block();
  // 两个块内 barrier：front 守护"正在计算"的缓冲区，back 守护"正在加载"
  // 的缓冲区。二者在每轮迭代后互换角色（见循环末尾的指针交换）
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
  auto frontBarrierPtr = &frontBarrier;
  auto backBarrierPtr = &backBarrier;
  if (block.thread_rank() == 0) {
    // 期望到达数 = 块内线程数（所有线程都会参与 arrive）
    init(&frontBarrier, block.size());
    init(&backBarrier, block.size());
  }
  __syncthreads(); // 确保 barrier 初始化完成后再使用

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 本线程所在的线程束在"线程块分块"中的位置
  const uint warpIdx = threadIdx.x / WARPSIZE; // 本线程属于第几个线程束
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // 线程束分块内的子分块尺寸
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // 本线程在"线程束子分块"中的位置
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // 双缓冲：共享内存分配两份空间，容量是单缓冲的两倍
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // 让 A/B 指针指向本线程块负责的分块，C 指向本线程束负责的输出分块
  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算本线程要往共享内存搬运的元素索引（float4 向量化：每步 4 个元素）
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // 累加结果缓存在寄存器中，避免频繁读写共享内存
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // 按线程束分块粒度缓存的 A/B 片段（寄存器级缓存）
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // 当前使用哪块缓冲区（0 或 1），每轮迭代翻转
  int As_offset = 0;
  int Bs_offset = 0;

  // 双缓冲流水线：先把第 0 个分块异步拷入缓冲 0
  loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As + As_offset * BM * BK, Bs + Bs_offset * BK * BN, innerRowA,
      innerColA, innerRowB, innerColB, (*frontBarrierPtr));

  // 最外层循环：每轮推进一个 K 分块
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // 双缓冲：异步拷入"下一个"分块（缓冲 (1-offset)），
    // 与下面的计算（缓冲 offset）并行进行，互不等待
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, As + (1 - As_offset) * BM * BK,
        Bs + (1 - Bs_offset) * BK * BN, innerRowA, innerColA, innerRowB,
        innerColB, (*backBarrierPtr));

    // 等当前分块的异步拷贝全部完成，然后计算它
    //（arrive_and_wait：等所有线程到达 + 所有拷贝完成）
    (*frontBarrierPtr).arrive_and_wait();
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As + As_offset * BM * BK,
        Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;     // 指向右侧的下一个 K 分块
    B += BK * N; // 指向下方的下一个 K 分块

    // 翻转缓冲区：下一轮"正在计算"的变成刚才"正在加载"的
    As_offset = 1 - As_offset;
    Bs_offset = 1 - Bs_offset;
    // 交换 front/back barrier 的角色：front 继续守护"即将被计算"的缓冲区，
    // back 继续守护"即将被加载"的缓冲区
    auto tmp = frontBarrierPtr;
    frontBarrierPtr = backBarrierPtr;
    backBarrierPtr = tmp;

    __syncthreads(); // 保证所有人完成角色交换后再进入下一轮
  }

  // 循环结束时还剩最后一个分块：等它就绪后计算
  (*frontBarrierPtr).arrive_and_wait();
  processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      regM, regN, threadResults, As + As_offset * BM * BK,
      Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
      threadColInWarp);

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
