#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# ─── 文件路径 ──────────────────────────────────────────────────────────
MD_CSV    = '/scratch/lgong1/finalproject/pems_data/step32_md.csv'
EXTER_CSV = '/scratch/lgong1/finalproject/pems_data/step31_fillExter.csv'

# ─── 1. 读取 Mahalanobis 结果 ───────────────────────────────────────────
md = pd.read_csv(MD_CSV, parse_dates=['timestamp'])

# 整体异常比例
overall_pct = md['mask_md'].mean() * 100
print(f"整体异常比例: {overall_pct:.2f}%\n")

# ─── 2. 读取外部标记以重建 group ────────────────────────────────────────
usecols = ['station_id','timestamp','is_weekend','is_holiday','in_custom_event']
flags = pd.read_csv(EXTER_CSV, usecols=usecols, parse_dates=['timestamp'])
flags['group'] = (
    flags['is_holiday'].astype(int) * 4 +
    flags['in_custom_event'].astype(int) * 2 +
    flags['is_weekend'].astype(int) * 1
)

# ─── 3. 合并并按 group 统计 ────────────────────────────────────────────
df = md.merge(
    flags[['station_id','timestamp','group']],
    on=['station_id','timestamp'],
    how='left'
)

print("各组异常比例:")
print((df.groupby('group')['mask_md'].mean() * 100).sort_index(), "\n")

# ─── 4. 绘制 md_squared 分布（含阈值）──────────────────────────────────
for grp, sub in df.groupby('group'):
    plt.hist(sub['md_squared'], bins=200, alpha=0.5, label=f'Group {grp}')
    # 可选：画出自适应阈值线 (取 95% 分位)
    thr = sub['md_squared'].quantile(0.95)
    plt.axvline(thr, linestyle='--', label=f'95%ile (G{grp})')

plt.yscale('log')
plt.legend()
plt.title('Mahalanobis $d^2$ by Group')
plt.xlabel('md_squared')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step33_checkMD.py 
整体异常比例: 1.59%

各组异常比例:
group
0    1.565250
1    1.650582
Name: mask_md, dtype: float64 


进程已结束，退出代码为 0"""