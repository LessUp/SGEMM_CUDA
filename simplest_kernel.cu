// simplest_kernel.cu —— 最小 CUDA 示例：理解 grid / block / thread 的索引映射
// 学习价值：CUDA 的一切都从"线程如何编号"开始。
// 本示例启动 1 个线程块、16 个线程（block_size = 4*4），
// 每个线程按 threadIdx.x 拆成 (x, y) = (threadIdx.x/4, threadIdx.x%4)，
// 往矩阵 A 的 (x, y) 位置写 x，往 B 的 (x, y) 位置写 y。
// 运行后打印的 4×4 网格里，每格显示 [行号|列号]，
// 直观展示了二维坐标与一维线程索引之间的换算关系。
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 每个线程负责 (x, y) 一个元素：
// threadIdx.x 是 0..15 的一维编号，把它拆成二维 (x, y)
__global__ void kernel(uint *A, uint *B, int row) {
  auto x = threadIdx.x / 4; // 前 4 个线程是一行（x 固定、y 变化）
  auto y = threadIdx.x % 4; // 余数决定列
  A[x * row + y] = x;
  B[x * row + y] = y;
}

int main(int argc, char **argv) {
  uint *Xs, *Ys;
  uint *Xs_d, *Ys_d;

  uint SIZE = 4; // 4×4 矩阵

  Xs = (uint *)malloc(SIZE * SIZE * sizeof(uint));
  Ys = (uint *)malloc(SIZE * SIZE * sizeof(uint));

  cudaMalloc((void **)&Xs_d, SIZE * SIZE * sizeof(uint));
  cudaMalloc((void **)&Ys_d, SIZE * SIZE * sizeof(uint));

  dim3 grid_size(1, 1, 1);     // 1 个线程块
  dim3 block_size(4 * 4);      // 每块 16 个线程（一维编号 0..15）
  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, 4);

  cudaMemcpy(Xs, Xs_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ys, Ys_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  // 打印结果矩阵，每格 [X|Y] 对应线程写入的 (x, y)
  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  cudaFree(Xs_d);
  cudaFree(Ys_d);
  free(Xs);
  free(Ys);
}
