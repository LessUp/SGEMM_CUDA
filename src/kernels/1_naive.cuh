#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * Kernel 1：sgemm_naive —— 朴素实现（基线）
 * ============================================================================
 *
 * 作用
 * ----
 * 最直观的 SGEMM 写法：每个线程负责计算输出矩阵 C 中的一个元素，C 的
 * 尺寸是 M×N，所以一共启动 M×N 个线程，一一对应。
 *
 * 线程到矩阵元素的映射
 * --------------------
 * 线程网格（grid）的 x 方向覆盖 C 的行（M），y 方向覆盖 C 的列（N）：
 *   x = blockIdx.x * blockDim.x + threadIdx.x
 *   y = blockIdx.y * blockDim.y + threadIdx.y
 * 即"blockIdx 定位哪个线程块、threadIdx 定位块内哪个线程"的经典换算。
 *
 * 为什么性能差（本 kernel 存在的意义）
 * -----------------------------------
 * 1. 每个输出元素要沿 K 维做一次长度为 K 的点积，K 次全局内存读取，
 *    数据完全没有复用——同一线程读入的每个元素只参与一次乘加就
 *    被丢弃（A 的每个元素本可被 N 个线程共用，这里却各读各的）；
 * 2. 相邻线程读取的 A 列（x 变化）地址不连续，全局内存访问无法合并
 *    （coalescing），一条 32 字节的访存请求可能只用到 4 字节；
 * 3. 后面所有 kernel 的优化，本质上都是围绕"让数据被复用、让访存
 *    连续"这两点展开的。本 kernel 是性能对比的 0 号基准。
 *
 * 代码细节
 * --------
 * - x < M && y < N 的边界判断：当矩阵尺寸不是线程块尺寸的整数倍时，
 *   多启动的线程会被这里拦下来，防止越界写内存。
 * ============================================================================
 */

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // 当前线程负责的 C 元素坐标：x 是行、y 是列
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // 越界检查：矩阵尺寸未必是线程块尺寸的整数倍，多余线程直接退出
  if (x < M && y < N) {
    float tmp = 0.0;
    // 沿 K 维做点积：C[x][y] = sum(A[x][k] * B[k][y])
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α·(A@B) + β·C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
