#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize 不是 constexpr，这里用常量代替

/*
 * ============================================================
 * kernel 10：线程束分块（Warp Tiling）版 SGEMM
 * ============================================================
 *
 * 【作用】
 * 在共享内存分块（tiling）的基础上，把每个线程块负责的 BM×BN 的 C 子块
 * 继续划分：每个线程束（warp）独立计算一个 WM×WN 的连续子块，线程束内部
 * 再按 WMITER×WNITER 个子分块逐步计算。本文件导出 kernel 函数
 * sgemmWarptiling。
 *
 * 【解决的核心性能问题】
 * 1. 共享内存（shared memory）的存储体冲突（bank conflict）：
 *    上一版 kernel 中，每个线程直接负责一个 TM×TN 的结果子块，同一线程束
 *    内相邻线程从共享内存取数时地址间隔 TM 个 float（本配置为 4），16 个
 *    线程就扫完 32 个存储体（bank），形成 8 路冲突，取数延迟放大数倍。
 *    线程束分块后，线程束内线程同一时刻读的是同一行中"相同或相邻"的位置
 *    （见 processFromSmem 的加载模式：每 4 个线程一组读同一地址，即广播，
 *    组与组之间地址连续），从根本上消除冲突。
 * 2. 寄存器级数据复用：regM/regN 从共享内存读出一次，可参与 TM×TN 次
 *    乘加运算，把共享内存的读取次数降为原来的 1/(TM×TN)，缓解共享内存
 *    带宽压力。
 *
 * 【相对上一版 kernel 的改进】
 * - 上一版以线程为单位组织计算，共享内存加载存在多路 bank 冲突；
 * - 本版以线程束为单位组织计算，配合"转置存储 A 分块"（见下），加载阶段
 *   做到无 bank 冲突；且线程束之间互不等待，每个 K 分块只需同步两次。
 *
 * 【关键概念讲解】
 * - 为什么 A 分块要转置后存入共享内存？
 *   全局内存（global memory）中 A 的 K 维是连续存储的。若原样存入，As 中
 *   同一 K 列的各行地址间隔 BK 个 float，线程束内线程取数时地址跨度过大，
 *   必然冲突。转置后 As 按"列优先"布局（As[col * BM + row]），K 维成为
 *   连续方向，线程束内线程沿行取数时地址连续、自然满足合并访问
 *   （coalescing）。
 * - 为什么 B 分块不用转置？
 *   B 的加载与计算都沿 N 方向（行内）连续进行，Bs[row * BN + col] 的布局
 *   天然满足连续访问。
 * - 同步点：每个 K 分块只有两次 __syncthreads（加载完成后、计算完成后各
 *   一次）。因为各线程束只读写自己负责的 C 区域，彼此无数据依赖，不需要
 *   更细粒度的同步。
 * - 寄存器压力：每个线程的累加器 threadResults 有 WMITER×TM×WNITER×TN =
 *   64 个 float，加上 regM/regN 各 8 个，共 80 个寄存器/线程。一旦超出硬件
 *   上限（255 个），多余部分会溢出到局部内存，性能急剧下降——这也是后续
 *   kernel 要控制分块尺寸的原因。
 * ============================================================
 */

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 下面这段被注释掉的是"内联汇编版"的加载写法，效果与上面的 float4
    // 完全相同，仅作参考：ld.global.nc.v4.f32 一次读回 4 个 float。
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    // 写回共享内存时完成转置：As 按列优先（K 方向连续）存放
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // B 不需要转置，按原布局整体搬入（float4 一次拷 4 个元素）
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // 内联汇编版（仅作参考）：
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
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

} // namespace wt

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
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
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

  // 为当前线程块分块在共享内存中分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 让 A/B 指针指向本线程块负责的分块：A 的行起点、B 的列起点
  A += cRow * BM * K;
  B += cCol * BN;
  // 让 C 指针指向本线程束负责的输出分块
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算本线程要往共享内存搬运的元素索引：
  // 每次用 128 位（float4）搬运，即每线程每步 4 个元素
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

  // 最外层循环：沿 K 维遍历每个线程块分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads(); // 等待本分块全部搬入共享内存
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // 指向右侧的下一个 K 分块
    B += BK * N; // 指向下方的下一个 K 分块
    __syncthreads(); // 等所有线程用完共享内存，下一轮才能覆盖
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
