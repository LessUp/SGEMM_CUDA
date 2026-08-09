// sgemm.cu —— 主程序：跑基准测试（benchmark）并验证正确性
// 用法: ./sgemm <kernel 编号>（0 表示 cuBLAS，1-12 对应 12 个优化 kernel）
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "请选择一个 kernel（范围 0 - 12，0 表示 NVIDIA cuBLAS）"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // 获取 kernel 编号
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "请输入合法的 kernel 编号（0-12）" << std::endl;
    exit(EXIT_FAILURE);
  }

  // 从环境变量读取要使用的 GPU 设备号（默认为 0 号设备）
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("正在设备 %d 上运行 kernel %d。\n", kernel_num, deviceIdx);

  // 打印设备信息（调试用，默认注释掉）
  // CudaDeviceInfo();

  // 创建 cuBLAS 句柄（handle）。cublasCreate 返回 cublasStatus_t
  // 类型的值，用来判断句柄是否创建成功（0 表示成功）。
  // 之后所有 cuBLAS 调用都要传入这个句柄。
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "创建 cuBLAS 句柄失败。" << std::endl;
    exit(EXIT_FAILURE);
  };

  // 用 cudaEvent 对 GPU 上的执行做计时：
  // cudaEvent 相当于在目标 stream 里发布一个"事件标记"，
  // 记录两个事件之间的墙钟时间，比 CPU 计时更准确。
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS 的算力上限要到 8192 尺寸才打满，这里只测到 4096
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "最大矩阵尺寸: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM 的标量参数，C = α·(A@B) + β·C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // 主机（CPU）侧矩阵
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // 设备（GPU）侧矩阵

  // 分配主机内存，一次性按最大尺寸分配，基准测试时对不同尺寸
  // 只用其中左上角的部分（所有矩阵都是方阵 m=n=k=size）
  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  // 分配设备显存并把数据拷贝过去
  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50; // 每个尺寸重复跑 50 次取平均，消除波动
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "矩阵尺寸(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // 先用 cuBLAS 算一份标准答案（dC_ref），再用我们的 kernel 算（dC），
    // 对比验证正确性。验证也执行一次正式 kernel，避免冷启动误差。
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // 执行目标 kernel，修改结果矩阵
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // 检查 kernel 运行期间是否有异步错误
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout << "与 NVIDIA cuBLAS 对比未能通过正确性验证。" << std::endl;
        if (m <= 128) {
          // 小矩阵出错时，把输入/输出矩阵存盘方便排查
          std::cout << " 将错误的输出记录到 " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A（输入矩阵）:\n";
          print_matrix(A, m, n, fs);
          fs << "B（输入矩阵）:\n";
          print_matrix(B, m, n, fs);
          fs << "C（kernel 的输出）:\n";
          print_matrix(C, m, n, fs);
          fs << "标准答案（cuBLAS 输出）:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    // 正式计时：在 GPU stream 上记录起始事件，跑完 repeat_times 次后记录结束事件
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // 重复跑时不重置 dC（每次都会重新覆盖），省去一次拷贝开销
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // 换算成秒

    // 矩阵乘法的计算量是 2*m*n*k 次浮点运算（m*n 个输出元素，
    // 每个元素做 k 次乘加，乘和加各算一次浮点运算）
    long flops = 2 * m * n * k;
    printf(
        "平均耗时: (%7.6f) s, 性能: (%7.1f) GFLOPS. 尺寸: (%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // 基准跑完一轮后把 dC 恢复成 dC_ref 的内容
    // （跑我们自己的 kernel 时 dC 被改写了，下一轮验证还需要它）
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // 释放 CPU 和 GPU 内存
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
