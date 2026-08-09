#!/usr/bin/env bash

# 注意：这里只用了 set -u（未定义变量时报错），没用 set -e，
# 因为某个配置跑基准失败不应中断整个调优过程，跳过继续即可。

# =============================================================================
# 内核 11（warp 级分块 + 双缓冲版）参数搜索脚本
#
# 与 kernel_10 的 autotuner 完全同构：枚举参数组合 -> 筛掉非法组合 ->
# sed 改源码常量 -> make 重编 -> 跑基准记录结果。
# 唯一区别是内核 11 在 10 的基础上加了双缓冲（double buffering）：
# 同时申请两份共享内存缓冲，加载下一块数据时计算当前块，
# 让访存与计算重叠，隐藏全局内存延迟。搜索空间与检查条件完全一样。
#
# 参数含义：
#   BM/BN：每个线程块（block）负责的输出子块大小（M x N 方向）
#   BK   ：内层循环每次迭代沿 K 方向推进的长度（共享内存 tile 的 K 维度）
#   WM/WN：每个 warp 负责的输出子块大小（M x N 方向）
#   WN_ITER：warp 沿 N 方向把 WN 分成几段迭代计算
#   TM/TN：每个线程最终负责的输出元素个数（M x N 方向）
#   NUM_THREADS：每个 block 的线程数（这里从 128 / 256 里选）
#
# warp 级分块的核心思想（kernel 10/11 与 kernel 5/9 的关键区别）：
# block 先把输出按 BM x BN 切块，再按 WM x WN 分给各 warp，
# 每个 warp 内部再循环迭代，让每个线程算 TM x TN 个元素。
# 因此参数必须满足层层整除关系（见下面的 if 检查），
# 否则会出现 warp 无法完整覆盖子块、寄存器溢出或访存越界。
# =============================================================================

# 定义各参数的候选值（这就是被枚举的搜索空间）
BK_VALUES=(8 16 32 64)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
NUM_THREADS_VALUES=(128 256)

# 进入 build 目录（无论从哪个目录调用脚本，相对路径都成立）
cd "$(dirname "$0")"
cd "../build"

# 要被 sed 修改参数的源码文件（常量定义所在位置）
RUNNER="../src/runner.cu"
# 调优过程日志：每个配置的基准结果都会追加到这里
OUTPUT="../benchmark_results/kernel_11_autotune_results.txt"

# 清空旧日志，从零开始记录
echo "" > $OUTPUT

# 指定使用哪块 GPU（多卡机器上需要，单卡可不管）
export DEVICE="0"
# GPU 的 warp 大小固定为 32 个线程（NVIDIA GPU 的硬件特性）
WARPSIZE=32


# 计算总配置数（各参数候选数量的乘积），用于显示"第几组/共几组"
TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]} ))"
CONFIG_NUM=0

# 嵌套循环枚举所有参数组合（即笛卡尔积）
for BK in "${BK_VALUES[@]}"; do
for BM in "${BM_VALUES[@]}"; do
for BN in "${BN_VALUES[@]}"; do
for WM in "${WM_VALUES[@]}"; do
for WN in "${WN_VALUES[@]}"; do
for WN_ITER in "${WNITER_VALUES[@]}"; do
for TM in "${TM_VALUES[@]}"; do
for TN in "${TN_VALUES[@]}"; do
for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
echo ""
CONFIG_NUM=$(( CONFIG_NUM + 1 ))
# ---- 前置条件检查：不满足的组合直接跳过 ----
NUM_WARPS=$(( NUM_THREADS / 32 ))
# 1) block 的输出子块必须能被 warp 子块完整划分（不能有剩余）
if ! (( BN % WN == 0 && BM % WM == 0 )); then
  echo "错误：BN % WN 必须为 0 且 BM % WM 必须为 0（block 子块必须能被 warp 子块完整划分）。"
  continue
fi
# 2) 划分出的 warp 个数必须恰好等于实际的 warp 数（线程数 / 32）
if ! (( (BN / WN) * (BM / WM) == NUM_WARPS )); then
  echo "错误：(BN / WN) * (BM / WM) 必须等于 NUM_WARPS（warp 数量对不上）。"
  continue
fi
# 3) warp 子块 WM*WN 必须能被"32 线程 x TM x TN x WN_ITER"整除，
#    保证每个 warp 内部迭代时线程的工作量能完整分配
if ! (( (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )); then
  echo "错误：(WM * WN) % (WARPSIZE * TM * TN * WN_ITER) 必须为 0（warp 内部工作量无法完整分配）。"
  continue
fi
# 4) 由上面整除关系推出每次迭代的趟数，再检查 WM/WN 能否被趟数划分
WM_ITER=$(( (WM * WN) / (WARPSIZE * TM * TN * WN_ITER) ))
if ! (( WM % WM_ITER == 0 && WN % WN_ITER == 0 )); then
  echo "错误：WM % WM_ITER 必须为 0 且 WN % WN_ITER 必须为 0（迭代趟数划分失败）。"
  continue
fi
# 5) 共享内存 tile 按 float4（4 个元素）向量化加载的整除条件
if ! (( (NUM_THREADS * 4) % BK == 0 )); then
  echo "错误：(NUM_THREADS * 4) % BK 必须为 0（向量化加载会跨 tile 边界）。"
  continue
fi
if ! (( (NUM_THREADS * 4) % BN == 0 )); then
  echo "错误：(NUM_THREADS * 4) % BN 必须为 0（向量化加载会跨 tile 边界）。"
  continue
fi
# 6) 量化计算（每线程一次算 16 个元素）的整除条件
if ! (( BN % (16 * TN) == 0 )); then
  echo "错误：BN 必须是 16 * TN 的整数倍（量化块无法完整放入）。"
  continue
fi
if ! (( BM % (16 * TM) == 0 )); then
  echo "错误：BM 必须是 16 * TM 的整数倍（量化块无法完整放入）。"
  continue
fi
# 7) 整块共享内存 tile 加载的向量化整除条件
if ! (( (BM * BK) % (4 * NUM_THREADS) == 0 )); then
  echo "错误：(BM * BK) % (4 * NUM_THREADS) 必须为 0（A tile 加载越界）。"
  continue
fi
if ! (( (BN * BK) % (4 * NUM_THREADS) == 0 )); then
  echo "错误：(BN * BK) % (4 * NUM_THREADS) 必须为 0（B tile 加载越界）。"
  continue
fi

# 用 sed 把当前组合的参数写进源码里的常量定义
# （正则匹配 "const uint K11_XXX = ..." 整行并替换成新值）
sed -i "s/const uint K11_NUM_THREADS = .*/const uint K11_NUM_THREADS = $NUM_THREADS;/" $RUNNER
sed -i "s/const uint K11_BN = .*/const uint K11_BN = $BN;/" $RUNNER
sed -i "s/const uint K11_BM = .*/const uint K11_BM = $BM;/" $RUNNER
sed -i "s/const uint K11_BK = .*/const uint K11_BK = $BK;/" $RUNNER
sed -i "s/const uint K11_WM = .*/const uint K11_WM = $WM;/" $RUNNER
sed -i "s/const uint K11_WN = .*/const uint K11_WN = $WN;/" $RUNNER
sed -i "s/const uint K11_WNITER = .*/const uint K11_WNITER = $WN_ITER;/" $RUNNER
sed -i "s/const uint K11_TM = .*/const uint K11_TM = $TM;/" $RUNNER
sed -i "s/const uint K11_TN = .*/const uint K11_TN = $TN;/" $RUNNER

# 重新编译（make 检测到源码变化，只重编受影响的目标）
make

# 打印当前配置（同时追加进日志文件）
echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$BK BM=$BM BN=$BN WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN NUM_THREADS=$NUM_THREADS" |& tee -a $OUTPUT
# 运行基准测试（sgemm 11 = 11 号内核）
# timeout 8 秒兜底：某些非法配置可能让内核挂死，8 秒没跑完直接杀掉继续下一个
timeout -v 8 ./sgemm 11 | tee -a $OUTPUT
done
done
done
done
done
done
done
done
done
