// runner.cu —— 所有 kernel 的"启动器"（launcher）
// 每个 runXxx 函数负责：确定分块尺寸（BM/BN/BK/TM/TN 等模板参数）、
// 计算网格（grid）和线程块（block）的维度、然后启动对应的 kernel。
// 这里的模板参数即"分块方案"，是性能调优的核心旋钮。
#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

// 统一检查 CUDA 调用是否出错：出错时打印出错位置并终止程序
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA 错误] 位置 %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

// 打印当前 GPU 设备的关键硬件参数（调试用）
void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("设备 ID: %d\n\
    名称: %s\n\
    计算能力（Compute Capability）: %d.%d\n\
    显存位宽 memoryBusWidth: %d\n\
    每线程块最大线程数 maxThreadsPerBlock: %d\n\
    每 SM 最大线程数 maxThreadsPerMultiProcessor: %d\n\
    每线程块最大寄存器数 maxRegsPerBlock: %d\n\
    每 SM 最大寄存器数 maxRegsPerMultiProcessor: %d\n\
    全局显存 totalGlobalMem: %zuMB\n\
    每线程块共享内存 sharedMemPerBlock: %zuKB\n\
    每 SM 共享内存 sharedMemPerMultiprocessor: %zuKB\n\
    常量内存 totalConstMem: %zuKB\n\
    SM 数量 multiProcessorCount: %d\n\
    线程束大小 Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

// 用随机数填充矩阵（数值范围 [-5, 5) 附近，包含正负）
void randomize_matrix(float *mat, int N) {
  // 注意：这里用 gettimeofday 而不是 srand((unsigned)time(NULL))，
  // 因为 time() 的精度只有秒级，多次快速调用会生成相同的随机数序列
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

// 把矩阵填充为 i（按索引递增），用于验证索引映射是否正确
void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("复制失败: 第 %d 个元素，共 %d 个元素。\n", i, N);
}

// 把矩阵打印成 MATLAB 风格的格式，方便对照检查
void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // 设置浮点精度和定点格式
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // 设置字段宽度并写值
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

// 验证 kernel 输出与参考结果（cuBLAS 输出）是否一致，
// 允许 0.01 的误差（浮点运算顺序不同会带来微小差异）
bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > 0.01) {
      printf("结果发散! 期望值 %5.2f, 实际值 %5.2f (差值 %5.2f) 在位置 %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

// 向上取整除法：numerator/denominator，有余数则进位
int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

// 调用 cuBLAS 的 fp32 全精度 GEMM 作为参考实现/性能基线
void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS 使用列主序（column-major）。我们代码里是行主序（row-major），
  // 所以交换 A、B 的传入顺序：因为 (B^T * A^T)^T = A * B，
  // 把矩阵转置信息编码进参数顺序即可，不用真的转置数据。
  // 这里用全 fp32 精度计算
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// cuBLAS 混合精度（bf16）模式：乘法的操作数被降精度到 bf16 再算，
// 速度约为 fp32 的 4 倍（Tensor Core 加速），但精度有损失
void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// cuBLAS TF32 模式：另一种混合精度（Tensor Core），精度介于 fp32 和 bf16 之间
void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ===== 以下每个函数启动一个 kernel，参数含义：=====
// BM/BN: 线程块负责的 C 子块尺寸（行×列）
// BK:    K 维每次循环加载的分块大小
// TM/TN: 每个线程负责的 C 子块尺寸（1 号、2 号 kernel 无此概念）
// grid = (N/BN, M/BM)，block = 块内线程数
// -------------------------------------------------

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  // 每线程块 32×32 个线程，每个线程负责 C 中一个元素
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32); // 一维线程块，线程索引拆成 (tx, ty) 用
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // 这个 kernel 只通过共享内存访问全局内存，L1 缓存不再被利用，
  // 所以把 L1 的容量尽可能划给共享内存（carveout 配置）。
  // 目前这个改动没有实际收益（占用率受寄存器和线程数限制），
  // 但这是一个值得知道的调优手段。
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64; // 线程块负责 64×64 的 C 子块
  const uint BN = 64;
  const uint BK = 8; // 每次沿 K 维加载 8 行
  const uint TM = 8; // 每个线程负责 8 个 C 元素（一行）
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM); // 64*64/8 = 512 个线程
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8; // 每线程负责 TM×TN 的 C 子块
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128; // 大矩阵用大块：128×128
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN)); // 128*128/(8*8) = 256 个线程
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 小矩阵退回 64×64 的分块——这是对 kernel 缺少边界检查
    // 这个根本问题的权宜之计（hack）
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 同上：小矩阵时的权宜方案
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 同上：小矩阵时的权宜方案
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 同上：小矩阵时的权宜方案
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // 下面两套参数是自动调优脚本（scripts/kernel_9_autotuner.sh）
  // 在不同 GPU 上搜索出的最优分块方案
  // A100 上的最优参数（搜索结果是 BK=16, TM=4, TN=4, BM=64, BN=64）
  // const uint K9_BK = 16;
  // const uint K9_TM = 4;
  // const uint K9_TN = 4;
  // const uint K9_BM = 64;
  // const uint K9_BN = 64;
  // A6000 上的最优参数
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);

  // 以下 static_assert 都是"分块一致性"约束：
  // 保证每轮循环从全局内存往共享内存搬数据时，
  // 所有线程恰好铺满一个完整分块（不会只搬某一行的一部分，产生"量子化"误差），
  // 并且 float4 向量化加载时对齐正确。
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 必须是 K9_BK 的倍数，避免 GMEM->SMEM 分块时的量子化问题"
      "（每轮迭代只加载 Bs 最后一行的部分元素）");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 必须是 K9_BN 的倍数，避免 GMEM->SMEM 分块时的量子化问题"
      "（每轮迭代只加载 As 最后一行的部分元素）");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN 必须是 16*K9_TN 的倍数，避免量子化效应");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM 必须是 16*K9_TM 的倍数，避免量子化效应");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK 必须是 4*256 的倍数，才能做向量化加载");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK 必须是 4*256 的倍数，才能做向量化加载");

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // A100 上的最优参数
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // A6000 上的最优参数
  // 说明：warptiling 在 blocktiling 之上再加一层"warp 分块"——
  // 每个线程块先分成若干 warp，每个 warp 负责 WM×WN 的 C 子块；
  // 参数含义：WM/WN = warp 子块尺寸，WNITER = warp 沿 N 维迭代次数
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warp 子块必须正好铺满线程块子块
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  // warp 数量必须等于 (BN/WN)*(BM/WM)
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // 每个 warp 子块要能被线程数整除
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // 迭代划分必须整除
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  // 以下 static_assert 与 kernel 9 相同：保证全局内存→共享内存
  // 加载时没有量子化问题、向量化加载对齐正确
  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 必须是 K9_BK 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 Bs 最后一行的部分元素）");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 必须是 K9_BN 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 As 最后一行的部分元素）");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN 必须是 16*TN 的倍数，避免量子化效应");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM 必须是 16*TM 的倍数，避免量子化效应");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK 必须是 4*256 的倍数，才能做向量化加载");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK 必须是 4*256 的倍数，才能做向量化加载");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // A100 上的最优参数
  // const uint K11_NUM_THREADS = 256;
  // const uint K11_BN = 128;
  // const uint K11_BM = 64;
  // const uint K11_BK = 16;
  // const uint K11_WN = 32;
  // const uint K11_WM = 32;
  // const uint K11_WNITER = 2;
  // const uint K11_TN = 4;
  // const uint K11_TM = 4;
  // A6000 上的最优参数
  // 说明：这个 kernel 在 warptiling 基础上加了双缓冲——
  // 共享内存分成两块（偶数/奇数轮次交替使用），
  // 计算第 i 轮时预先异步加载第 i+1 轮的数据，隐藏访存延迟。
  const uint K11_NUM_THREADS = 256;
  const uint K11_BN = 256;
  const uint K11_BM = 128;
  const uint K11_BK = 16;
  const uint K11_WN = 32;
  const uint K11_WM = 128;
  const uint K11_WNITER = 1;
  const uint K11_TN = 8;
  const uint K11_TM = 8;
  dim3 blockDim(K11_NUM_THREADS);

  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  // warp 子块必须正好铺满线程块子块
  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  // warp 数量必须等于 (BN/WN)*(BM/WM)
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // 每个 warp 子块要能被线程数整除
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // 迭代划分必须整除
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  // 双缓冲版：一半线程负责预取下一轮的数据（NUM_THREADS/2 参与），
  // 所以下面的整除条件里用的是 NUM_THREADS/2
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 必须是 BK 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 Bs 最后一行的部分元素）");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 必须是 BN 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 As 最后一行的部分元素）");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN 必须是 16*TN 的倍数，避免量子化效应");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM 必须是 16*TM 的倍数，避免量子化效应");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK 必须是 4*256 的倍数，才能做向量化加载");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK 必须是 4*256 的倍数，才能做向量化加载");

  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  sgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
                       K11_TM, K11_TN, K11_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  // A6000 上的最优参数
  // 说明：这是 11 号 kernel 的变体，双缓冲的实现方式不同
  //（主要区别在于异步加载的同步机制，可对照两个文件学习）
  const uint K12_NUM_THREADS = 128;
  const uint K12_BN = 128;
  const uint K12_BM = 128;
  const uint K12_BK = 16;
  const uint K12_WN = 64;
  const uint K12_WM = 64;
  const uint K12_WNITER = 4;
  const uint K12_TN = 4;
  const uint K12_TM = 8;
  dim3 blockDim(K12_NUM_THREADS);

  constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

  // warp 子块必须正好铺满线程块子块
  static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
  // warp 数量必须等于 (BN/WN)*(BM/WM)
  static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

  // 每个 warp 子块要能被线程数整除
  static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
                0);
  constexpr uint K12_WMITER =
      (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
  // 迭代划分必须整除
  static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

  // 与 kernel 10 相同的分块一致性约束
  static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
                "NUM_THREADS*4 必须是 K9_BK 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 Bs 最后一行的部分元素）");
  static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
                "NUM_THREADS*4 必须是 K9_BN 的倍数，避免 GMEM->SMEM 分块时的"
                "量子化问题（每轮迭代只加载 As 最后一行的部分元素）");
  static_assert(K12_BN % (16 * K12_TN) == 0,
                "BN 必须是 16*TN 的倍数，避免量子化效应");
  static_assert(K12_BM % (16 * K12_TM) == 0,
                "BM 必须是 16*TM 的倍数，避免量子化效应");
  static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BM*BK 必须是 4*256 的倍数，才能做向量化加载");
  static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BN*BK 必须是 4*256 的倍数，才能做向量化加载");

  dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
  runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
                           K12_TM, K12_TN, K12_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// 统一入口：根据 kernel 编号分发到对应的启动函数
void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("未知的 kernel 编号");
  }
}
