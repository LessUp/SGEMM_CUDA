#!/usr/bin/env bash

# 开启"出错即退出 + 管道错误检测"：某个内核崩溃时立即报错退出，
# 避免脚本"跑完了"但结果不完整却没被发现
set -euo pipefail

# =============================================================================
# 全量基准测试脚本
#
# 流程：
#   1. 依次运行 ./build/sgemm 0 ~ 10 号内核（每个内核内部会遍历多组矩阵规模），
#      用 tee 把控制台输出同时打印到屏幕并保存为文本文件
#      benchmark_results/<内核编号>_output.txt
#   2. 全部跑完后调用绘图脚本 plot_benchmark_results.py：
#      它解析这些文本文件，画出各内核的性能对比图（benchmark_results.png），
#      并把 4096 规模的性能对比表回写进 README.md
#
# 注意：11 号内核（双缓冲版本）不在循环范围内，需要时把 {0..10} 改成 {0..11}。
# =============================================================================

# 结果目录（不存在则创建）
mkdir -p benchmark_results

# 依次跑 0~10 号内核。内核编号是 sgemm 的命令行参数，
# 每个内核会打印类似下面格式的结果行，绘图脚本靠正则解析它们：
#   Average elapsed time: (0.005661) s, performance: (24277.4) GFLOPS. size: (4096).
for kernel in {0..10}; do
    echo ""
    ./build/sgemm $kernel | tee "benchmark_results/${kernel}_output.txt"
    # 跑完一个内核稍作停顿再跑下一个，避免高负载下测量值互相干扰
    sleep 2
done

# 解析结果、画对比图并更新 README
python3 plot_benchmark_results.py
