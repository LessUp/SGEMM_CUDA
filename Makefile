# =============================================================================
# 项目根 Makefile —— 对 CMake 的薄封装，把常用操作缩成一个命令
#
# 常用用法：
#   make                     # Release 编译（等价于 make build）
#   make debug               # Debug 编译（带设备端调试信息，性能差，仅调试用）
#   make clean               # 删除 build 目录，彻底重新编译
#   make bench               # 跑全量基准测试并生成对比图（调用 gen_benchmark_results.sh）
#   make profile KERNEL=10 PREFIX=v1_
#                            # 用 Nsight Compute 剖析 10 号内核，导出报告加 v1_ 前缀
#   make cuobjdump           # 反汇编 warptiling 内核的 SASS / PTX，输出到 build/ 下
# =============================================================================

# 声明伪目标：这些名字不代表真实文件，make 不会因为"同名文件已存在"而跳过它们
.PHONY: all build debug clean profile bench cuobjdump

# 使用的 cmake 命令（系统里装了其他版本 cmake 时，改这一处即可）
CMAKE := cmake

# 构建输出目录：CMake 生成的 Makefile 与全部中间产物都在这（编译慢多半是缓存问题，
# make clean 后重编即可）
BUILD_DIR := build
# 基准测试结果输出目录（profile 报告的导出位置）
BENCHMARK_DIR := benchmark_results

# 默认目标：直接编译 Release 版本
all: build

# ---- Release 编译 ----
# 流程：建目录 -> cmake 生成构建系统（Release 开启 O3 等全部优化）-> make 真正编译
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

# ---- Debug 编译 ----
# Debug 版会给设备代码附加 -G 选项（见 CMakeLists.txt）：保留符号、关闭优化，
# 方便用 cuda-gdb / Nsight 单步跟踪内核执行，但性能会差很多。
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

# ---- 清理 ----
# 删除整个 build 目录。CMake 的缓存有时会"记错"文件列表
# （比如新增/删除源文件后），遇到奇怪报错先 clean 再重新编译。
clean:
	@rm -rf $(BUILD_DIR)

# ---- 反汇编 SASS / PTX ----
# 先查 warptiling 内核的符号名（nvcc 会对内核名字做 name mangling，
# 直接写名字搜不到），再反汇编成 SASS（GPU 机器码）和 PTX（中间表示），
# 最后用 c++filt 把 mangled 名字还原成可读形式。
# 看 SASS 是理解"编译器到底生成了什么指令"的最直接方式，
# 也是排查性能问题时（比如寄存器溢出、多余的边界检查）的利器。
FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# ---- 单内核性能剖析 ----
# 用法：make profile KERNEL=<内核编号 0~11> PREFIX=<文件名前缀，可选>
# 调用 Nsight Compute（ncu）采集全量指标（occupancy、访存带宽、指令数、
# 存储体冲突等），报告导出到 benchmark_results/<前缀>kernel_<编号>.ncu-rep，
# 可用 Nsight Compute GUI 打开查看。剖析时必须指定内核编号，
# 否则 ncu 不知道要分析 sgemm 的哪一个 kernel。
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

# ---- 全量基准测试 ----
# 依次跑 0~10 号内核，把日志存进 benchmark_results/，最后自动生成对比图
bench: build
	@bash gen_benchmark_results.sh
