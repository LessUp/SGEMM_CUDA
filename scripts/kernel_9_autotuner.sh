#!/usr/bin/env bash

# 注意：这里只用了 set -u（未定义变量时报错），没用 set -e，
# 因为某个配置跑基准失败不应中断整个调优过程，跳过继续即可。

# =============================================================================
# 内核 9（自动调优版）参数搜索脚本
#
# 内核 9 的分块参数是编译期常量（写在 src/runner.cu 和
# src/kernels/9_kernel_autotuned.cuh 里），而最优参数在不同 GPU 上不一样，
# 手工一个个试很费劲，所以写脚本自动搜索：
#
#   1. 枚举下面定义的所有参数组合（BK x TM x TN x BM x BN，NUM_THREADS 固定 256）
#   2. 跳过不满足前置条件的组合（下面 6 个 if 检查的整除性约束，
#      不满足的话要么编译报错、要么共享内存/寄存器溢出、要么向量化加载越界）
#   3. 用 sed 把参数值写进源码里的常量定义
#   4. make 重新编译
#   5. 运行 ./sgemm 9 跑基准，把结果追加进 benchmark_results/kernel_9_autotune_results.txt
#
# 参数含义：
#   BM/BN：每个线程块（block）负责的输出子块大小（M 方向 x N 方向）
#   BK   ：内层循环每次迭代沿 K 方向推进的长度（也是共享内存 tile 的 K 维度）
#   TM/TN：每个线程负责的输出元素个数（M 方向 x N 方向）
# 这些参数共同决定共享内存用量、寄存器用量和每个线程的访存/计算比率，
# 直接影响占用率（occupancy）和数据复用率，是 SGEMM 调优的核心旋钮。
# =============================================================================

# 定义各参数的候选值（这就是被枚举的搜索空间）
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)

# 进入 build 目录（无论从哪个目录调用脚本，相对路径都成立）
cd "$(dirname "$0")"
cd "../build"

# 要被 sed 修改参数的源码文件（常量定义所在位置）
RUNNER="../src/runner.cu"
KERNEL="../src/kernels/9_kernel_autotuned.cuh"
# 调优过程日志：每个配置的基准结果都会追加到这里
OUTPUT="../benchmark_results/kernel_9_autotune_results.txt"

# 清空旧日志，从零开始记录
echo "" > $OUTPUT

# 指定使用哪块 GPU（多卡机器上需要，单卡可不管）
export DEVICE="2"

# 计算总配置数（各参数候选数量的乘积），用于显示"第几组/共几组"
TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# 嵌套循环枚举所有参数组合（即笛卡尔积）
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            # 跳过不满足前置条件的组合（否则会编译报错或运行越界）
            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
            # 条件 1：共享内存 tile 按 float4（4 个元素）向量化加载时，
            # 要求"线程数 x 4"能整除 BK / BN，否则加载会跨过 tile 边界
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "向量化检查：跳过 $config，因为 (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
              continue
            fi
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "向量化检查：跳过 $config，因为 (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
              continue
            fi
            # 条件 2：量化计算（每个线程一次算 16 个元素）要求
            # BN / BM 能被 16*TN / 16*TM 整除，保证量化块能完整放进去
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "量化检查：跳过 $config，因为 BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
              continue
            fi
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "量化检查：跳过 $config，因为 BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
              continue
            fi
            # 条件 3：整块共享内存 tile 的加载也要能被 4 个元素一组向量化
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "向量化检查：跳过 $config，因为 (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "向量化检查：跳过 $config，因为 (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # 用 sed 把当前组合的参数写进源码里的常量定义
            # （正则匹配 "const uint K9_XXX = ..." 整行并替换成新值）
            sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
            sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
            sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
            sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
            sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
            sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL

            # 重新编译（make 检测到源码变化，只重编受影响的目标）
            make

            # 打印当前配置（同时追加进日志文件）
            echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |& tee -a $OUTPUT
            # 运行基准测试（sgemm 9 = 9 号内核）
            # timeout 4 秒兜底：某些非法配置可能让内核挂死，4 秒没跑完直接杀掉继续下一个
            timeout -v 4 ./sgemm 9 | tee -a $OUTPUT
          done
        done
      done
    done
  done
done
