# =============================================================================
# 共享内存存储体（bank）分布计算脚本
#
# 背景知识：GPU 的共享内存在硬件上被分成 32 个存储体（bank），
# 每个存储体每周期只能返回一个地址的数据（4 字节）。
# 当一个 warp（32 个线程）同时访问共享内存时，若多个线程访问的是
# 同一个 bank 的不同地址，就发生"存储体冲突（bank conflict）"——
# 硬件被迫串行处理，访问被拆成多轮，相当于白白损失带宽。
#
# 本脚本模拟 kernel 7 / kernel 8 里的访问模式：
# 一个 warp 的 32 个线程读取一块 16x16 的共享内存 tile
# （256 个 float，每个线程读 8 个），逐个计算每个元素落在哪个 bank，
# 统计冲突情况，对比两种内存布局：
#   1. banks_naive     ：每行 32 个 float（行宽 = 32，无填充）
#   2. banks_one_extra ：每行 33 个 float（每行末尾多填 1 个 float）
#
# 运行：python3 scripts/bank_calc.py
# =============================================================================

# 两种布局的 bank 编号计算公式：
# 元素位于第 r 行、第 c 列，其字节地址 = (r * 行宽 + c) * 4，
# bank 编号 = 地址 / 4 % 32 = (r * 行宽 + c) % 32。
# 行宽 32 时：(r*32+c)%32 恒等于 c%32，即同一列的元素全部挤在同一个 bank 上
# （下面运行结果可见：16 路冲突，只有 2/32 个 bank 被访问）；
# 行宽 33 时：每换一行 bank 编号整体 +1，行与行之间互相错开，
# 冲突从 16 路大幅降到 2 路。这就是 kernel 8 用"每行填充一个 float"
# 消除存储体冲突的原理。
# 注意：脚本只统计每个线程 8 个元素中"起始位置"的 bank 分布（简化模拟），
# 实际内核里 256 个元素全部参与统计，冲突消除得更彻底。
banks_naive = lambda r, c: (r * 32 + c) % 32
banks_one_extra = lambda r, c: (r * 33 + c) % 32

# 每个线程读取的元素个数（16x16 tile = 256 个 float，32 个线程平分）
ITEMS_PER_WARP = 8


def printBankConflicts(bank_fun):
    """
    模拟一个 warp 的访问，打印每个线程的 bank 分布与冲突统计。

    线程编号 i（0~31）与 tile 中元素的对应关系：
        row = (i * ITEMS_PER_WARP) // 16   —— 所在行
        col = (i * ITEMS_PER_WARP) % 16    —— 所在列
    即每 16 个线程负责一行（16 个 float），32 个线程正好覆盖 16x16 的 tile。
    """
    for c in range(1):
        banks = []
        for i in range(32):
            row = (i * ITEMS_PER_WARP) // 16
            col = (i * ITEMS_PER_WARP + c) % 16
            banks.append((i, row, col, bank_fun(row, col)))
        # 打印 32 个线程各自的 (线程号, 行, 列, bank 号)
        print("步骤", c, "\n", "\n".join(["(" + ",".join(str(x) for x in i) + ")" for i in banks]))
        # 统计每个 bank 被多少个线程命中
        d = {k: 0 for k in range(32)}
        for i in banks:
            d[i[-1]] += 1

        # 统计一共访问到了多少个不同的 bank
        count = 0
        for key, val in d.items():
            if val > 0:
                count += 1

        # 冲突数 = 被访问次数最多的那个 bank 的命中次数：
        # 例如某 bank 被 16 个线程同时访问，硬件就要拆成 16 轮传输，冲突数即为 16。
        print(
            f"存储体冲突数（步骤 {c}）：{sorted(d.items(), key=lambda item: item[1], reverse=True)[0][1]}，访问到的存储体数：{count}/32\n"
        )


# 对比两种布局：注意第二个参数原本是想传"行宽"，但行宽已编码在
# 上面的 bank 计算公式（闭包）里，因此这里不再传（原脚本传了第二个
# 参数会导致 TypeError，已修正）
print("---朴素布局（每行 32 个 float，无填充）---")
printBankConflicts(banks_naive)

print("\n---偏移布局（每行 33 个 float，多填充 1 个）---")
printBankConflicts(banks_one_extra)
