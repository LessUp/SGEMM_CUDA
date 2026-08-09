#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * Kernel 4：sgemm1DBlocktiling —— 一维分块（1D blocktiling）
 * ============================================================================
 *
 * 作用
 * ----
 * 每个线程块负责计算输出矩阵 C 中一个 BM×BN 的子块。做法是：先把 A、B
 * 中本块需要的数据分块（tile）搬进共享内存（shared memory），再沿 K 维
 * 以 BK 为步长滑动，逐块累加出完整的子块。每个线程只负责"一行"里的
 * TM 个连续结果——结果是一维排布的，所以叫一维分块；这 TM 个结果缓存在
 * 寄存器（register）数组里。
 *
 * 解决的核心性能问题
 * ------------------
 * 此前的朴素 kernel 让每个线程直接从全局内存（global memory）读 A、B
 * 的元素，相邻线程读取的数据大量重叠——同一个数被反复从延迟数百个周期
 * 的全局内存里取，全局内存的带宽和延迟成了瓶颈。
 *
 * 本 kernel 的思路：把数据复用从"反复访问全局内存"变成"共享内存里的
 * 一次搬运 + 多次命中"。每个数据只从全局内存读一次，之后全部命中共享
 * 内存（片上存储，带宽和延迟都比全局内存好一个数量级）。在分块内部，
 * 每个 A 元素被 BN 个输出元素的计算复用，每个 B 元素被 BM 个输出复用，
 * 全局内存的读取量因此降为原来的约 1/BN、1/BM。
 *
 * 相对上一个 kernel 的改进
 * ------------------------
 * 1. 引入共享内存分块：这是"分块（tiling）"思想的核心——把大矩阵切成
 *    能装进共享内存的小块逐块计算，让数据复用发生在片上，而不是靠
 *    L2 缓存兜底。
 * 2. 用寄存器数组 threadResults[TM] 累积结果：累加过程完全在寄存器里
 *    进行，不碰内存；一个分块算完后统一写回 C，写回时顺便完成
 *    alpha/beta 的线性组合。
 * 3. 把 K 维的点积循环放在最外层：内层循环的每次迭代都用同一个 Bs 值
 *    （threadCol 固定的那一列），先把它读进临时变量 tmpB，内层 TM 次
 *    乘加（FMA）共享这一个数——Bs 的读取从 TM×BK 次降为 BK 次。
 *
 * 关键概念：寄存器、共享内存与全局内存的分工
 * ------------------------------------------
 * 这是 CUDA 优化的主线——数据越"热"（复用次数越多），越要放在快的
 * 存储里：
 *   - 寄存器：每线程私有、零延迟，但容量小（每线程约 255 个，实际
 *     还要给编译器留余量），只能存本线程自己的数据；
 *   - 共享内存：线程块内共享的片上存储，延迟约 20~30 个周期，读写
 *     之间需要用 __syncthreads() 同步；
 *   - 全局内存：容量最大、延迟数百周期，只应承担"搬运"职责。
 *
 * 访问模式说明
 * ------------
 * - 全局内存加载按 threadIdx.x 线性展开（连续线程对应连续地址），
 *   满足合并访问（coalescing）：线程束（warp）一次请求即可取回一整段
 *   连续数据，32 个标量访问被合并成几次大块传输；
 * - 读共享内存 Bs 时，线程束内 threadCol 连续、地址连续，无存储体
 *   冲突（bank conflict）；读 As 时同一线程束的线程行号相同、访问同一
 *   地址，属于广播（broadcast），同样高效；
 * - blockIdx.x/y 的分配是刻意的：让编号连续的线程块顺序访问 B 的列、
 *   共享 A 的同一行。实测对调 x/y 会让大矩阵的性能下降约 30%——
 *   顺序访问 B 的列时空间局部性更好，L2 缓存命中率更高。
 *
 * 前提约束
 * --------
 * 加载与计算共用同一批线程：assert 要求 BM*BK == blockDim.x 且
 * BN*BK == blockDim.x（即 BM == BN），此时每个线程恰好加载一个 A 元素
 * 和一个 B 元素。TODO：可改为让每个线程加载多个元素，更好地利用各级
 * 缓存。
 * ============================================================================
 */

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // 这里的 blockIdx 分配是刻意为之：翻转 x 和 y 会让大矩阵的性能下降
  // 约 30%。当前（更快）的配置确保编号连续的线程块顺序访问 B 的列、
  // 共享 A 的同一行；翻转后则共享 A 的列，但访问 B 时地址不连续。
  // 因此当前配置的空间局部性更好，L2 缓存命中率更高。
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 每个线程束（warp，32 个线程）共计算 32×TM 个元素，32 是列方向的
  // 线程数
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // 在共享内存中为当前分块分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 把指针移动到本线程块负责的 A 行、B 列的起点
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // TODO: 改为让每个线程加载多个元素，更好地利用各级缓存
  //（当前每个线程只加载一个元素）
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // 线程束级全局内存合并访问
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // 线程束级全局内存合并访问
  const uint innerRowB = threadIdx.x / BN;

  // 在寄存器文件中为结果分配线程私有的缓存
  float threadResults[TM] = {0.0};

  // 外层循环：沿 K 维逐个处理分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 把当前分块从全局内存搬运到共享内存
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // 指针前进到下一个 K 分块
    A += BK;
    B += BK * N;

    // 计算本线程的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 把点积循环放在外层：这样 Bs 的同一个元素在内层循环中会被复用，
      // 只需从共享内存读一次，缓存在临时变量里。
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // 把结果写回全局内存
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
