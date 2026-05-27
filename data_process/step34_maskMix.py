#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step34_maskMix.py

功能：
 1. 读取三种掩码：
    - 逻辑掩码（step30_logic_mask_continuous.csv）
    - Mahalanobis 掩码（step32_md.csv）
    - 日健康因子掩码（day_health_factor.csv）
 2. 将它们合并到一张表，生成最终的 mask_mix：
    timestamp, station_id, mask_logic, mask_md, mask_hf
 3. 输出 CSV 和 Parquet，供后续插值脚本按需加载

用法：
    cd finalproject/data_process
    python step34_maskMix.py

依赖：
    pip install pandas pyarrow fastparquet

    /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_maskMix.py
1. 读取逻辑掩码…
2. 读取 Mahalanobis 掩码…
3. 合并逻辑 & MD 掩码…
4. 读取日健康因子并生成 mask_hf…
5. 将日期信息映射到每条记录，并合并 mask_hf…
6. 重排列，输出 CSV & Parquet…
✔ 已保存掩码合并文件：
  CSV -> /scratch/lgong1/finalproject/pems_data/step34_maskMix.csv
  Parquet -> /scratch/lgong1/finalproject/pems_data/step34_maskMix.parquet

"""
import os
import pandas as pd

# ——— 参数 & 路径 ———
HF_THRESH    = 0.3  # 日健康因子阈值
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pems_data'))
LOGIC_CSV    = os.path.join(BASE_DIR, 'step30_logic_mask_continuous.csv')
MD_CSV       = os.path.join(BASE_DIR, 'step32_md.csv')
HF_CSV       = os.path.join(BASE_DIR, 'step20_day_health_factor_GPU.csv')
OUT_CSV      = os.path.join(BASE_DIR, 'step34_maskMix.csv')
OUT_PARQUET  = os.path.join(BASE_DIR, 'step34_maskMix.parquet')

def main():
    print("1. 读取逻辑掩码…")
    logic = pd.read_csv(
        LOGIC_CSV,
        usecols=['timestamp','station_id','mask_logic'],
        parse_dates=['timestamp']
    )

    print("2. 读取 Mahalanobis 掩码…")
    md = pd.read_csv(
        MD_CSV,
        usecols=['timestamp','station_id','mask_md'],
        parse_dates=['timestamp']
    )

    print("3. 合并逻辑 & MD 掩码…")
    df = logic.merge(
        md,
        on=['timestamp','station_id'],
        how='left'
    )
    # 缺失值视为 False
    df['mask_md'] = df['mask_md'].fillna(False)

    print("4. 读取日健康因子并生成 mask_hf…")
    hf = pd.read_csv(
        HF_CSV,
        parse_dates=['date']
    )
    # 如果列不是 'health_factor'，重命名最后一列
    if 'health_factor' not in hf.columns:
        hf = hf.rename(columns={hf.columns[-1]: 'health_factor'})
    hf['mask_hf'] = hf['health_factor'] < HF_THRESH
    hf = hf[['station_id','date','mask_hf']]

    print("5. 将日期信息映射到每条记录，并合并 mask_hf…")
    # 提取日期
    df['date'] = df['timestamp'].dt.date
    # 合并
    df = df.merge(
        hf.assign(date=hf['date'].dt.date),
        on=['station_id','date'],
        how='left'
    )
    df['mask_hf'] = df['mask_hf'].fillna(False)
    # 删除辅助列
    df.drop(columns=['date'], inplace=True)

    print("6. 重排列，输出 CSV & Parquet…")
    df = df[['timestamp','station_id','mask_logic','mask_md','mask_hf']]
    df.to_csv(OUT_CSV, index=False)
    # Parquet 文件方便分块、高效读取
    df.to_parquet(OUT_PARQUET, index=False, compression='snappy')

    print(f"✔ 已保存掩码合并文件：\n  CSV -> {OUT_CSV}\n  Parquet -> {OUT_PARQUET}")

if __name__ == '__main__':
    main()
