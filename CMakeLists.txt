# =============================================================================
# SGEMM 优化教程项目的 CMake 构建配置
# =============================================================================
# 构建流程：cmake 生成构建系统 -> make 实际编译。
# 本文件负责把所有 .cu 源文件编译成可执行文件，并指定 CUDA 架构与编译选项。
#
# 常见命令：
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 生成 Release 构建系统
#   cmake --build build                              # 编译
#   （项目根目录的 Makefile 已封装上述步骤，直接 make 即可）
# =============================================================================

# 要求 cmake 版本不低于 3.19（支持 CUDA 语言特性所需的最低版本）
cmake_minimum_required(VERSION 3.19)

# 声明项目：LANGUAGES 告诉 cmake 本项目源文件包含 C++ 和 CUDA，
# 这样 cmake 才会调用 nvcc 编译 .cu 文件，而不是把它们当普通 C++ 处理。
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)

# 生成 compile_commands.json：记录每个源文件的完整编译命令，
# clangd / VS Code 等编辑器靠它实现准确的代码跳转和静态检查。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找 CUDA 工具包（nvcc、cudart 运行时、cuBLAS 等库的头文件路径与库文件）
find_package(CUDA REQUIRED)

# 二次确认当前编译器支持 CUDA 语言（结果存在 CMAKE_CUDA_COMPILER 中，
# 若找不到 nvcc 会直接报错，方便在配置阶段就发现问题）
include(CheckLanguage)
check_language(CUDA)

# 使用 C++20 标准（项目代码用到了较新的标准库特性）
set(CMAKE_CXX_STANDARD 20)

# CUDA 计算能力（Compute Capability）：目标 GPU 的架构代号。
# 86 = sm_86 = Ampere 架构，对应 RTX 30 系列（如 RTX 3060Ti / 3080 等）。
# 编译出的 SASS（GPU 机器码）只针对该架构优化；换 GPU 时需改成对应值，
# 例如 RTX 40 系列（Ada）是 89，RTX 20 系列（Turing）是 75，
# 写错架构会导致"架构不支持"或性能劣化（退化为 PTX 即时编译）。
set(CUDA_COMPUTE_CAPABILITY 86)

# Debug 模式下的设备端（GPU）代码编译选项：
#   -G           ：生成设备端调试信息，同时关闭绝大多数优化（性能大幅下降，
#                  仅供 cuda-gdb / Nsight 单步调试内核时使用）
#   -src-in-ptx  ：把源代码行号信息嵌入 PTX，便于把 SASS 指令映射回 .cu 源码行
# 这里用了 CMake 的生成器表达式（$<...>），效果是"仅当构建类型为 Debug
# 且当前编译的是 CUDA 代码时才附加这两个选项"，保证 Release 构建不受影响。
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# 调试利器：想查看 ptxas 的寄存器用量 / 共享内存占用 / 溢出警告时，
# 取消下面这行的注释重新编译即可（会打印在编译输出里）：
# add_compile_options("--ptxas-options=-v")

# 头文件搜索路径：CUDA 工具包的头文件 + 项目自己的 src/ 目录
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# 自动收集 src/ 目录下所有源文件，放入变量 SRC 供后续使用
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# 目标 1：本项目主程序 —— 内含 0~11 号全部内核与基准测试入口，
# 用法：./build/sgemm <内核编号>，例如 ./build/sgemm 5
add_executable(sgemm sgemm.cu ${SRC})
# 指定该目标针对上面定义的架构（sm_86）编译
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# 链接 CUDA 运行时库（cudart）和 cuBLAS 库（0 号内核需要调用 cuBLAS 做对照）
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# 目标 2：单独跑 cuBLAS 基准的小程序（只看官方库的性能上限时用）
add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# 目标 3：最简内核演示程序（入门对照用，只依赖 CUDA 运行时，不需要 cuBLAS）
add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
