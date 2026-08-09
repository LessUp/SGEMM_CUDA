#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * Kernel 6：sgemmVectorize —— 向量化访问（vectorized memory access）
 * ============================================================================
 *
 * 作用
 * ----
 * 在 kernel 5（二维分块）的基础上，把全局内存与共享内存之间的搬运
 * 全部改为 float4（128 位）向量化（vectorized）访问：一条指令搬 4 个
 * float；同时把 A 转置后存入共享内存，消除读 As 时的存储体冲突
 * （bank conflict）。
 *
 * 解决的核心性能问题
 * ------------------
 * 1. 指令条数过多：kernel 5 中每个线程每搬一个 float 就要发一条访存
 *    指令。指令发射（instruction issue）带宽是有限的资源，大量访存
 *    指令会挤占计算指令的发射槽。改用 float4 后访存指令数降为 1/4，
 *    同样多的指令搬运了 4 倍的数据。
 * 2. 全局内存带宽利用率不足：全局内存按 32 字节扇区、128 字节缓存行
 *    传输。标量访问时每个线程只用 4 字节，缓存行的其余部分常被浪费；
 *    float4 让每个线程一次拿满 16 字节、线程束（warp）一次拿 512
 *    字节，请求数更少、每次传输更饱满，实际带宽利用率更高。
 * 3. 存储体冲突：kernel 5 读 As 时，相邻线程的地址步长是 32（存储体
 *    数）的整数倍，会发生存储体冲突。本 kernel 在写入共享内存时把
 *    A 转置（As 布局从 [BM][BK] 变为 [BK][BM]），读的时候相邻线程的
 *    步长变为 TM 个 float，落在不同存储体，冲突消失。
 *
 * 相对上一个 kernel 的改进
 * ------------------------
 * - A 的加载：float4 读全局内存，再逐分量写入 As（顺带完成转置）；
 * - B 的加载：float4 读全局、float4 写共享内存（B 无需转置，因为
 *   读 Bs 时本来就没有冲突）；
 * - 结果写回：C 的"读入 + alpha/beta 线性组合 + 写回"也用 float4
 *   完成，写回指令同样减少为原来的 1/4。
 *
 * 关键概念：为什么 float4 快？
 * ----------------------------
 * 硬件一次内存请求的最小有效单位远大于 4 字节（32 字节扇区、128 字节
 * 缓存行）。标量访问时，每个线程只用到一个 float，一次请求传回的
 * 大部分字节被浪费；float4 强制每个线程使用 16 字节对齐的整块，配合
 * 线程束内连续地址，让每次内存请求都被"用满"。指令少了、单次请求的
 * 效率高了，内存系统就不再是瓶颈。
 *
 * 前提：float4 要求 16 字节对齐——A/B/C 由 cudaMalloc 分配天然满足；
 * 共享内存数组基址由编译器对齐；模板尺寸（BK、BN 等）取 4 的倍数，
 * 保证换算后的地址偏移仍然对齐。
 *
 * 代价与权衡
 * ----------
 * 转置不是免费的：写 As 时各线程按转置后的位置写入，这一步本身存在
 * 少量存储体冲突；但写只发生每个分块一次，而读 As 在热点循环里要
 * 执行 BK×TM 次——用一次性的小代价换掉热点路径上的持续冲突，非常
 * 划算。
 *
 * 注意：转置后 As 的索引语义与 kernel 4/5 不同（读 As 用
 * As[dotIdx * BM + ...] 而非 As[... * BK + dotIdx]），阅读代码时请先
 * 弄清布局。
 * ============================================================================
 */

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // 每次加载 128 位 / 32 位 = 4 个元素
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // 在寄存器文件中为结果分配线程私有的缓存
  float threadResults[TM * TN] = {0.0};
  // As 和 Bs 的寄存器缓存：每个点积迭代把当前列读进寄存器
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 最外层循环：沿 K 维逐个处理分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 把当前分块从全局内存搬运到共享内存
    // 搬运 A 的同时进行转置：As 按 [BK][BM] 布局存放，
    // 这样计算阶段读取 As 时相邻线程的步长为 TM 个 float，
    // 落在不同存储体，消除存储体冲突。
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // B 保持 [BK][BN] 布局，直接整体拷贝即可
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 把 C 的对应 4 个元素以 float4 形式读入寄存器
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // 在寄存器中完成 alpha/beta 线性组合的更新
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
