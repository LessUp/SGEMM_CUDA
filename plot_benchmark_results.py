#!/usr/bin/env python3

# =============================================================================
# 基准测试结果绘图脚本
#
# 依赖：matplotlib / seaborn / pandas / tabulate（见 environment.yml 的 conda 环境）
#
# 用法：先运行 gen_benchmark_results.sh 生成 benchmark_results/<编号>_output.txt，
#       再运行本脚本（gen_benchmark_results.sh 也会自动调用它）。脚本会：
#         1. 解析每个 <内核编号>_output.txt，提取各矩阵规模下的 GFLOPS
#         2. 画"矩阵规模 vs 性能"的多折线对比图，保存为 benchmark_results.png
#         3. 生成 4096 规模下各内核相对 cuBLAS 的性能表，回写进 README.md
#
# 中文显示提示：若图里的中文标题显示成方块，说明系统缺少中文字体；
# 安装 Noto Sans CJK 等字体后，把下面的 rcParams["font.family"] 改成对应字体名即可。
# =============================================================================

import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pathlib import Path

# 设置统一的绘图风格，保证所有图外观一致：
matplotlib.style.use("fivethirtyeight")          # 基础风格（底色、网格线风格）
matplotlib.style.use("seaborn-v0_8-talk")        # 再叠加 seaborn 的 talk 语境（整体字号更大）
matplotlib.rcParams["font.family"] = "monospace" # 等宽字体：数字逐位对齐，刻度更好看
matplotlib.rcParams["figure.dpi"] = 200          # 高分辨率（200 DPI），图片放大不糊
plt.rcParams["savefig.facecolor"] = "white"      # 保存的图片底色用白色（默认可能带灰底）
# 备选方案（已注释，供尝试）：
# sn.set_theme(style="whitegrid", font_scale=1.0, rc={"figure.dpi": 200})
# plt.rcParams["font.family"] = "serif"           # 衬线字体（学术论文风格）
# sn.set_context("talk")

# 各内核编号对应的名称与优化要点（用于图例标注和 README 结果表）。
# 0~11 号内核是教程按优化步骤渐进式实现的 12 个版本：
#   0: cuBLAS               官方库基准（性能天花板，对照目标）
#   1: Naive                最朴素实现（无任何优化，性能最低）
#   2: GMEM Coalescing      全局内存合并访问（相邻线程访问相邻地址）
#   3: SMEM Caching         用共享内存缓存数据块，减少全局内存访问次数
#   4: 1D Blocktiling       一维分块：每个 block 处理一条输出"长条"
#   5: 2D Blocktiling       二维分块：每个 block 处理一个输出方块
#   6: Vectorized Mem Access   向量化（float4）内存访问，单条指令搬 4 个 float
#   7: Avoid Bank Conflicts (Linearize)  线性化索引消除存储体冲突
#   8: Avoid Bank Conflicts (Offset)     每行加填充偏移消除存储体冲突
#   9: Autotuning           自动调优：自动搜索最优分块参数组合
#   10: Warptiling          warp 级分块：每个 warp 负责一个输出子块
#   11: Double Buffering    双缓冲：数据预取与计算重叠，隐藏访存延迟
KERNEL_NAMES = {
    0: "cuBLAS",
    1: "Naive",
    2: "GMEM Coalescing",
    3: "SMEM Caching",
    4: "1D Blocktiling",
    5: "2D Blocktiling",
    6: "Vectorized Mem Access",
    7: "Avoid Bank Conflicts (Linearize)",
    8: "Avoid Bank Conflicts (Offset)",
    9: "Autotuning",
    10: "Warptiling",
    11: "Double Buffering",
}


def parse_file(file):
    """
    从单个基准测试输出文件中解析出 (矩阵规模, 性能) 数据。

    文件里每一行结果长这样（由 sgemm 程序打印）：
        Average elapsed time: (0.005661) s, performance: (24277.4) GFLOPS. size: (4096).

    正则表达式用三个括号把数字分别抓出来：
        group(1) -> 平均耗时（秒）
        group(2) -> 性能（GFLOPS：每秒十亿次浮点运算，越大越快）
        group(3) -> 矩阵规模（方阵边长，如 4096 表示 4096x4096）
    """
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {"size": [], "gflops": []}
    # 注意用 raw string（r"..."）：正则里的大量反斜杠是给正则引擎看的转义，
    # 不加 r 前缀 Python 会先做字符串转义并产生警告
    pattern = r"Average elapsed time: \((.*?)\) s, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        if r := re.match(pattern, line):  # 海象运算符：匹配成功时把结果赋给 r
            data["size"].append(int(r.group(3)))
            data["gflops"].append(float(r.group(2)))
    return data


def plot(df: pd.DataFrame):
    """
    画各内核的性能对比图。

    传入的 DataFrame 有 3 列：kernel（内核编号）、size（矩阵规模）、gflops（性能）。
    以矩阵规模为 x 轴、GFLOPS 为 y 轴，每个内核画一条折线，
    用 seaborn 的 lineplot 一次画出全部内核，方便直观对比优劣。
    """
    save_dir = Path.cwd()

    # 画布 18x10 英寸：图上有 12 条折线，画大一点才看得清
    plt.figure(figsize=(18, 10))
    # 用 husl 色板给每个内核分配一种颜色（颜色数随内核数量自动调整）
    colors = sn.color_palette("husl", len(df["kernel"].unique()))
    # 每个内核一条折线（lineplot 自动按 kernel 分组、按 size 排序连接）
    sn.lineplot(data=df, x="size", y="gflops", hue="kernel", palette=colors)
    # 再叠加散点，把实际测量点标出来（不重复显示图例）
    sn.scatterplot(data=df, x="size", y="gflops", hue="kernel", palette=colors, legend=False)

    # x 轴刻度就设在真实测过的规模上，而不是默认的均匀刻度
    plt.xticks(df["size"].unique())
    # 刻度文字旋转 45 度并右对齐，避免长数字互相遮挡
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # 在每条折线末尾旁标注"编号: 名称"，不用靠认颜色就能区分每条线
    for i, kernel in enumerate(df["kernel"].unique()):
        # 文字放在该内核测过的最后一个规模点、性能值上方 300 GFLOPS 处
        plt.text(
            df[df["kernel"] == i]["size"].iloc[-1],
            df[df["kernel"] == i]["gflops"].iloc[-1] + 300,
            f"{i}:{KERNEL_NAMES[i]}",
            color=colors[i],
            horizontalalignment="left",
            weight="medium",  # 中等字重，让标注更醒目
        )

    # 折线图自带图例与逐条标注重复，这里去掉图例
    plt.gca().get_legend().remove()

    plt.title("不同内核的性能对比（矩阵规模 vs GFLOPS）")
    plt.xlabel("矩阵规模（方阵边长）")
    plt.ylabel("GFLOPS/s（越高越快）")
    plt.tight_layout()  # 自动调整边距，防止坐标轴文字被裁掉

    # 图片保存在当前目录，文件名 benchmark_results.png
    plt.savefig(save_dir / "benchmark_results.png")


if __name__ == "__main__":
    # 结果目录：gen_benchmark_results.sh 生成的 <编号>_output.txt 都在这
    results_dir = Path("benchmark_results")
    assert results_dir.is_dir()  # 目录不存在说明还没跑过基准测试

    data = []
    for filename in results_dir.glob("*.txt"):
        # 文件名格式为 <内核编号>_output.txt，只处理这种文件
        if not filename.stem.split("_")[0].isdigit() and "_output" not in filename.stem:
            continue
        # 解析出该内核在所有规模下的性能
        results_dict = parse_file(filename)
        kernel_nr = int(filename.stem.split("_")[0])
        # 展平成"一行一条记录"的长表格式，便于 pandas 处理
        for size, gflops in zip(results_dict["size"], results_dict["gflops"]):
            data.append({"kernel": kernel_nr, "size": size, "gflops": gflops})
    df = pd.DataFrame(data)

    # 画性能对比图
    plot(df)

    # 生成"4096 规模对比表"并写回 README.md：
    # 只保留 4096 规模的记录，按性能升序排序；
    # 把内核编号替换成"编号: 名称"的形式；
    # 再以 cuBLAS 的性能为基准，算出每个内核的相对百分比。
    df = df[df["size"] == 4096].sort_values(by="gflops", ascending=True)[["kernel", "gflops"]]
    df["kernel"] = df["kernel"].map({k: f"{k}: {v}" for k, v in KERNEL_NAMES.items()})
    df["relperf"] = df["gflops"] / df[df["kernel"] == "0: cuBLAS"]["gflops"].iloc[0]
    df["relperf"] = df["relperf"].apply(lambda x: f"{x*100:.1f}%")
    df.columns = ["内核", "GFLOPs/s", "相对 cuBLAS 的性能"]

    # 把新表格写回 README.md：
    # README 里有一对 <!-- benchmark_results --> 注释标记，两者之间的内容
    # 就是上次生成的表格，用正则整段替换成新表格。
    with open("README.md", "r") as f:
        readme = f.read()
    # 删除旧结果（正则把两个标记之间的所有内容都匹配掉）
    readme = re.sub(
        r"<!-- benchmark_results -->.*<!-- benchmark_results -->",
        "<!-- benchmark_results -->\n{}\n<!-- benchmark_results -->".format(
            df.to_markdown(index=False)  # tabulate 把 DataFrame 转成 markdown 表格
        ),
        readme,
        flags=re.DOTALL,  # DOTALL 让 "." 也能匹配换行，保证能跨多行删除
    )
    # 写入新结果
    with open("README.md", "w") as f:
        f.write(readme)
