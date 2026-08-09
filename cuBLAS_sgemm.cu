// cuBLAS_sgemm.cu —— 独立小示例：演示如何用 cuBLAS 库做 SGEMM
// 学习价值：这是"调库"的写法，对比项目里"手写 kernel"的写法，
// 可以直观感受 cuBLAS 的接口约定（列主序！）与手动实现的区别。
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * 独立脚本：调用并基准测试标准 cuBLAS SGEMM 的性能
 */

int main(int argc, char *argv[]) {
  int m = 2; // A 是 m×k 矩阵
  int k = 3;
  int n = 4; // B 是 k×n 矩阵，C 是 m×n
  int print = 1;
  cudaError_t cudaStat;  // cudaMalloc 的状态
  cublasStatus_t stat;   // cuBLAS 函数的状态
  cublasHandle_t handle; // cuBLAS 上下文（句柄）

  int i, j;

  float *a, *b, *c;

  // 为主机端的 a、b、c 分配内存...
  a = (float *)malloc(m * k * sizeof(float));
  b = (float *)malloc(k * n * sizeof(float));
  c = (float *)malloc(m * n * sizeof(float));

  // 用 11 起的连续整数填充 A（m×k 个元素）
  int ind = 11;
  for (j = 0; j < m * k; j++) {
    a[j] = (float)ind++;
  }

  // 同样方式填充 B（k×n 个元素）
  ind = 11;
  for (j = 0; j < k * n; j++) {
    b[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < m * n; j++) {
    c[j] = (float)ind++;
  }

  // 设备（GPU）端指针
  float *d_a, *d_b, *d_c;

  // 为 d_a、d_b、d_c 分配显存...
  cudaMalloc((void **)&d_a, m * k * sizeof(float));
  cudaMalloc((void **)&d_b, k * n * sizeof(float));
  cudaMalloc((void **)&d_c, m * n * sizeof(float));

  stat = cublasCreate(&handle); // 初始化 CUBLAS 上下文

  cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);

  float alpha = 1.0f; // GEMM 公式：C = α·(A@B) + β·C
  float beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
    printf("A = (m×k: %d x %d)\n", m, k);
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        printf("%4.1f ", a[i * m + j]);
      }
      printf("\n");
    }
    printf("B = (k×n: %d x %d)\n", k, n);
    for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", b[i * n + j]);
      }
      printf("\n");
    }
    printf("C = (m×n: %d x %d)\n", m, n);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  // 关键点：cuBLAS 是列主序（column-major），而上面的矩阵是行主序存的。
  // 所以把 A、B 的维度参数调换传入（m×k 的 A 当成 k×m 的列主序矩阵，
  // 相当于对 A 做了一次"隐式转置"），结果 C 也是列主序读法。
  // 具体到 cublasSgemm 的签名：
  //   cublasSgemm(handle, opA, opB, 行数, 列数, K, alpha, A, lda, B, ldb, beta, C, ldc)
  // 这里传 (n, m, k) 作为输出矩阵的行列，配合 lda/ldb/ldc 参数完成转置映射。
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n,
                     d_a, k, &beta, d_c, n);

  cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  if (print == 1) {
    printf("\nSGEMM 之后的 C = \n");
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle); // 销毁 CUBLAS 上下文
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
