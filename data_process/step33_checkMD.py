#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# ─── filepath ──────────────────────────────────────────────────────────
MD_CSV    = '/scratch/lgong1/finalproject/pems_data/step32_md.csv'
EXTER_CSV = '/scratch/lgong1/finalproject/pems_data/step31_fillExter.csv'

# ─── 1. read Mahalanobis result ───────────────────────────────────────────
md = pd.read_csv(MD_CSV, parse_dates=['timestamp'])

# note
overall_pct = md['mask_md'].mean() * 100
print(f"note: {overall_pct:.2f}%\n")

# ─── 2. readnote group ────────────────────────────────────────
usecols = ['station_id','timestamp','is_weekend','is_holiday','in_custom_event']
flags = pd.read_csv(EXTER_CSV, usecols=usecols, parse_dates=['timestamp'])
flags['group'] = (
    flags['is_holiday'].astype(int) * 4 +
    flags['in_custom_event'].astype(int) * 2 +
    flags['is_weekend'].astype(int) * 1
)

# ─── 3. note group note ────────────────────────────────────────────
df = md.merge(
    flags[['station_id','timestamp','group']],
    on=['station_id','timestamp'],
    how='left'
)

print("note:")
print((df.groupby('group')['mask_md'].mean() * 100).sort_index(), "\n")

# ─── 4. note md_squared distribution(note)──────────────────────────────────
for grp, sub in df.groupby('group'):
    plt.hist(sub['md_squared'], bins=200, alpha=0.5, label=f'Group {grp}')
    # note: note (note 95% note)
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
note: 1.59%

note:
group
0    1.565250
1    1.650582
Name: mask_md, dtype: float64


process finished, exit codenote 0"""