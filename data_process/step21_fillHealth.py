#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step21_fillHealth_optimized.py

功能：
 - 读取原始长表 step1_raw_long.csv，按 chunk 逐块处理
 - 合并日级健康度表 day_health_factor.csv
 - 对于 health_conf 缺失的 station，填充值 0.0039
 - 输出合并结果到 step2_filled.csv
 - 显示进度条

"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── 配置 ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_LONG = BASE_DIR / "pems_data" / "step1_raw_long.csv"
DAY_HEALTH = BASE_DIR / "pems_data" / "step20_day_health_factor_GPU.csv"
OUT_FILE = BASE_DIR / "pems_data" / "step21_fillHealth.csv"

# 每个 chunk 的行数，可根据内存调整
CHUNKSIZE = 5_000_000
DEFAULT_HEALTH = 0.0039

# ─── 主函数 ─────────────────────────────────────────────────────────────
def main():
    # 读取日级健康度表
    dfh = pd.read_csv(DAY_HEALTH, parse_dates=['date'])

    # 计算总行数，用于进度条
    total_rows = sum(1 for _ in open(RAW_LONG, 'r', encoding='utf-8')) - 1
    reader = pd.read_csv(RAW_LONG, parse_dates=['timestamp'], chunksize=CHUNKSIZE)

    first_chunk = True
    with tqdm(total=total_rows, unit='rows', desc='Processing') as pbar:
        for chunk in reader:
            # 生成日期列
            chunk['date'] = chunk['timestamp'].dt.floor('D')
            # 合并健康度
            chunk = chunk.merge(dfh, on=['station_id','date'], how='left')
            # 缺失填默认
            chunk['health_conf'] = chunk['health_conf'].fillna(DEFAULT_HEALTH)
            # 写入输出
            chunk.to_csv(
                OUT_FILE,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False,
                encoding='utf-8'
            )
            first_chunk = False
            pbar.update(len(chunk))

if __name__ == '__main__':
    main()
