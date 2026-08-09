// runner.cuh —— 声明 runner.cu 中实现的工具函数
// 这些函数被主程序 sgemm.cu 使用：
// 矩阵初始化/拷贝/打印、正确性验证、计时、kernel 分发
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // 统一检查 CUDA 错误
void CudaDeviceInfo();    // 打印 GPU 设备信息

void range_init_matrix(float *mat, int N); // 用索引值填充矩阵
void randomize_matrix(float *mat, int N);  // 用随机数填充矩阵
void zero_init_matrix(float *mat, int N);  // 清零矩阵
void copy_matrix(const float *src, float *dest, int N); // 拷贝矩阵
void print_matrix(const float *A, int M, int N, std::ofstream &fs); // 打印矩阵到文件
bool verify_matrix(float *mat1, float *mat2, int N); // 对比两个矩阵是否一致（验证正确性）

float get_current_sec();                        // 获取当前时刻（微秒级）
float cpu_elapsed_time(float &beg, float &end); // 计算时间差（秒）

// 按 kernel 编号启动对应的 kernel（0 = cuBLAS，1-12 = 我们的实现）
void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);
