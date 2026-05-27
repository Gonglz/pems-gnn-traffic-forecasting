#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 自动调参与日级健康度生成脚本（GPU 加速版）

功能:
 1. 读取健康切片文件，为每个切片日期对所有站点打事件标签：
    - Good 映射预测值 li = +g_base
    - Bad  映射预测值 li = -1.0
 2. 构建全站点×日期日历，计算每个半衰期下的预测值 s(D)
 3. 在“中间态”和“事件日”分别计算 MSE，并按 alpha 加权得到混合损失
 4. 网格搜索半衰期，选出使混合损失最小的最佳半衰期
 5. 用最佳半衰期重新计算全量日级健康度 s(D)，映射至 [0,1] 输出 day_health_factor.csv
 6. 输出半衰期调优结果 step2_decay_tuning.csv

使用 CuPy 在 GPU 上加速矩阵运算。
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step20_healthWeigh_gpu.py
GPU 加速半衰期搜索: threshold=0.6, alpha=0.8
hl=0.1 天 → mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.2 天 → mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.3 天 → mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.5 天 → mse_mid=0.13566, mse_ext=0.00001, loss=0.10853
hl=0.7 天 → mse_mid=0.12731, mse_ext=0.00034, loss=0.10192
hl=1.0 天 → mse_mid=0.12859, mse_ext=0.00336, loss=0.10354
hl=5.0 天 → mse_mid=0.10552, mse_ext=0.15812, loss=0.11604
hl=7.0 天 → mse_mid=0.09231, mse_ext=0.24813, loss=0.12347
最佳半衰期: 0.3
已输出日级健康度表到 /scratch/lgong1/finalproject/pems_data/step20_day_health_factor_GPU.csv

进程已结束，退出代码为 0
"""

import pandas as pd
import numpy as np
import math
import cupy as cp
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────────────
PEMS_DIR       = Path(__file__).resolve().parent.parent / 'pems_data'
DET_DIR        = PEMS_DIR / 'pems_detector'
RAW_LONG       = PEMS_DIR / 'step1_raw_long.csv'
TUNE_CSV       = PEMS_DIR / 'step2_decay_tuning.csv'
HEALTH_CSV     = PEMS_DIR / 'step20_day_health_factor_GPU.csv'
# 避免出现 zero 半衰期，使用合理正数列表
half_life_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 5.0, 7.0]
g_base         = 0.9       # Good 冲击幅度
threshold      = 0.6       # 中间态阈值
alpha          = 0.8       # 中间态损失权重

# ─── 1. 构建事件表 ────────────────────────────────────────────────────
stations = pd.read_csv(RAW_LONG, usecols=['station_id'])['station_id'].astype(str).unique()
slice_dates, good_map = [], {}
for f in sorted(DET_DIR.glob('Detector Health *.xlsx')):
    date = pd.to_datetime(f.stem.split()[-1], format='%m%d%Y').date()
    slice_dates.append(date)
    dfh = pd.read_excel(f, sheet_name='Report Data')
    good_map[date] = set(dfh['VDS'].astype(str).str.strip())
# 事件表: station_id, slice_date_int, li
events = []
for date in slice_dates:
    day_int = np.datetime64(date, 'D').astype(int)
    for sid in stations:
        li = g_base if sid in good_map[date] else -1.0
        events.append((sid, day_int, li))
events_df = pd.DataFrame(events, columns=['station_id','day_int','li'])
events_df.sort_values(['station_id','day_int'], inplace=True)
events_df.reset_index(drop=True, inplace=True)

# ─── 2. 构建日历及 GPU 准备 ────────────────────────────────────────────
start_int = np.datetime64(min(slice_dates), 'D').astype(int)
end_int   = np.datetime64(max(slice_dates), 'D').astype(int)
days_range = np.arange(start_int, end_int+1)
# GPU 上的日期数组
days_gpu = cp.asarray(days_range)
# 预构建 station 事件映射
station_events = {sid: (grp['day_int'].values.astype(np.int32),
                        grp['li'].values.astype(np.float32))
                  for sid, grp in events_df.groupby('station_id', sort=False)}

# ─── 3. GPU 加速网格搜索半衰期（混合损失） ─────────────────────────────
results = []
print(f"GPU 加速半衰期搜索: threshold={threshold}, alpha={alpha}")
for hl in half_life_list:
    k = math.log(2) / hl
    mse_mid_sum, mid_count = 0.0, 0
    ext_errors = []
    for sid, (days_arr, li_arr) in station_events.items():
        days_evt = cp.asarray(days_arr)
        li_evt   = cp.asarray(li_arr)
        diffs = cp.abs(days_evt[:, None] - days_gpu[None, :]).astype(cp.float32)
        w     = cp.exp(-k * diffs)
        sum_w = w.sum(axis=0)
        s_all = (w * li_evt[:, None]).sum(axis=0) / sum_w
        s_cpu = cp.asnumpy(s_all)
        mid_mask = (np.abs(s_cpu) <= threshold)
        mse_mid_sum += np.sum(s_cpu[mid_mask]**2)
        mid_count   += mid_mask.sum()
        for d_int, li_val in zip(days_arr, li_arr):
            idx = int(d_int - start_int)
            ext_errors.append((s_cpu[idx] - float(li_val))**2)
    mse_mid = mse_mid_sum / mid_count if mid_count>0 else np.nan
    mse_ext = float(np.mean(ext_errors))
    loss    = alpha * mse_mid + (1 - alpha) * mse_ext
    print(f"hl={hl} 天 → mse_mid={mse_mid:.5f}, mse_ext={mse_ext:.5f}, loss={loss:.5f}")
    results.append((hl, mse_mid, mse_ext, loss))
# 保存调优结果
df_tune = pd.DataFrame(results, columns=['half_life','mse_mid','mse_ext','loss'])
df_tune['best'] = df_tune['loss']==df_tune['loss'].min()
df_tune.to_csv(TUNE_CSV, index=False)
# 不要将 best_hl 强制转为 int，保持原始浮点值
best_hl = df_tune.loc[df_tune['best'],'half_life'].iloc[0]
print(f"最佳半衰期: {best_hl}")

# ─── 4. GPU 生成日级健康度表 ──────────────────────────────────────────
k = math.log(2) / best_hl
records = []
for sid, (days_arr, li_arr) in station_events.items():
    days_evt = cp.asarray(days_arr)
    li_evt   = cp.asarray(li_arr)
    diffs = cp.abs(days_evt[:, None] - days_gpu[None, :]).astype(cp.float32)
    w     = cp.exp(-k * diffs)
    sum_w = w.sum(axis=0)
    s_all = (w * li_evt[:, None]).sum(axis=0) / sum_w
    s_cpu = cp.asnumpy(s_all)
    for i, d_int in enumerate(days_range):
        date_ts = pd.Timestamp(int(d_int), unit='D')
        health_conf = (s_cpu[i] + 1.0) / (g_base + 1.0)
        records.append((sid, date_ts, health_conf))
# 输出日级健康度
df_health = pd.DataFrame(records, columns=['station_id','date','health_conf'])
df_health.to_csv(HEALTH_CSV, index=False)
print("已输出日级健康度表到", HEALTH_CSV)
