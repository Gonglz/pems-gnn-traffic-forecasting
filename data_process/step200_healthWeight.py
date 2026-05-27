#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 自动调参与日级健康度生成脚本（增强版：中间态+事件日混合损失）

功能:
 1. 读取健康切片文件，为每个切片日期对所有站点打事件标签：
    - Good 映射预测值 li = +g_base
    - Bad  映射预测值 li = -1.0
 2. 构建全站点×日期日历，计算每个半衰期下的预测值 s(D)
 3. 在“中间态”和“事件日”分别计算 MSE，并按 alpha 加权得到混合损失
 4. 网格搜索半衰期，选出使混合损失最小的最佳半衰期
 5. 用最佳半衰期重新计算全量日级健康度 s(D)，映射至 [0,1] 输出 day_health_factor.csv
 6. 输出半衰期调优结果 step2_decay_tuning.csv

 /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step200_healthWeight.py
开始增强版半衰期搜索，threshold=0.60, alpha=0.80
hl=1天 → mse_mid=0.12859, mse_ext=0.00336, loss=0.10354
hl=2天 → mse_mid=0.16528, mse_ext=0.03900, loss=0.14002
hl=3天 → mse_mid=0.12071, mse_ext=0.08043, loss=0.11265
hl=5天 → mse_mid=0.10671, mse_ext=0.15812, loss=0.11699
hl=7天 → mse_mid=0.08838, mse_ext=0.24813, loss=0.12033
hl=10天 → mse_mid=0.05564, mse_ext=0.37557, loss=0.11962
hl=14天 → mse_mid=0.04474, mse_ext=0.49858, loss=0.13551
hl=21天 → mse_mid=0.03328, mse_ext=0.62205, loss=0.15104
最佳半衰期: 1
已输出日级健康度表到 /scratch/lgong1/finalproject/pems_data/day_health_factor.csv
浮点精度对齐
GPU 版里我们最开始用了 float32 做指数和加法，CPU 是 float64（NumPy 默认）。如果你想完全复现，GPU 上也可以改为 w = cp.exp(-k*diffs.astype(cp.float64)).astype(cp.float64)，或者直接用 float64 的数组。
进程已结束，退出代码为 0


输出:
 - pems_data/step2_decay_tuning.csv   (half_life, mse_mid, mse_ext, loss, best)
 - pems_data/day_health_factor.csv    (station_id, date, health_conf)
"""
import pandas as pd
import numpy as np
import math
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────────────
PEMS_DIR       = Path(__file__).resolve().parent.parent / 'pems_data'
DET_DIR        = PEMS_DIR / 'pems_detector'
RAW_LONG       = PEMS_DIR / 'step1_raw_long.csv'
TUNE_CSV       = PEMS_DIR / 'step2_decay_tuning.csv'
HEALTH_CSV     = PEMS_DIR / 'day_health_factor.csv'
half_life_list = [1, 2, 3, 5, 7, 10, 14, 21]
g_base         = 0.9       # Good 冲击幅度
threshold      = 0.6       # 中间态阈值
alpha          = 0.8       # 中间态损失权重

# ─── 1. 构建事件表 ────────────────────────────────────────────────────
# 读取全量站点列表
stations = pd.read_csv(RAW_LONG, usecols=['station_id'])['station_id'].astype(str).unique()
# 收集切片日期及对应 Good 站点集合
slice_dates = []
good_map = {}
for f in sorted(DET_DIR.glob('Detector Health *.xlsx')):
    date = pd.to_datetime(f.stem.split()[-1], format='%m%d%Y').date()
    slice_dates.append(date)
    dfh = pd.read_excel(f, sheet_name='Report Data')
    good_map[date] = set(dfh['VDS'].astype(str).str.strip())
# 生成事件表: station_id, slice_date, li
events = []
for date in slice_dates:
    goods = good_map[date]
    for sid in stations:
        li = g_base if sid in goods else -1.0
        events.append((sid, date, li))
events_df = pd.DataFrame(events, columns=['station_id','slice_date','li'])
events_df.sort_values(['station_id','slice_date'], inplace=True)
events_df.reset_index(drop=True, inplace=True)

# ─── 2. 构建日历 ─────────────────────────────────────────────────────
start_date = min(slice_dates)
end_date   = max(slice_dates)
dates = pd.date_range(start_date, end_date, freq='D').date
calendar = [(sid, d) for sid in stations for d in dates]

# ─── 3. 网格搜索半衰期（混合损失） ─────────────────────────────────────
results = []
print("开始增强版半衰期搜索，threshold=%.2f, alpha=%.2f" % (threshold, alpha))
for hl in half_life_list:
    k = math.log(2) / hl
    errors_mid = []  # |s|<=threshold 点与 0 的平方
    errors_ext = []  # 事件日点与真 li 的平方
    # 遍历日历
    for sid, D in calendar:
        sub = events_df[events_df.station_id == sid]
        if sub.empty:
            continue
        # 计算 s(D)
        dates_arr = sub['slice_date'].values
        li_arr    = sub['li'].values
        diffs = np.abs((dates_arr - D).astype('timedelta64[D]').astype(int))
        w     = np.exp(-k * diffs)
        s     = (w * li_arr).sum() / w.sum()
        # 判断
        if D in dates_arr:
            # 事件日，真标签为 li
            # 确保只取单个元素，避免数组转标量的警告
            mask    = dates_arr == np.datetime64(D)
            li_vals = li_arr[mask]
            true_li = float(li_vals[0]) if li_vals.size >= 1 else 0.0
            errors_ext.append((s - true_li) ** 2)
        elif abs(s) <= threshold:
            # 中间态，与 0 做差
            errors_mid.append(s ** 2)
        # 其它情况不纳入损失
    mse_mid = np.mean(errors_mid) if errors_mid else np.nan
    mse_ext = np.mean(errors_ext) if errors_ext else np.nan
    # 混合损失
    loss_mid = mse_mid if not np.isnan(mse_mid) else 0.0
    loss_ext = mse_ext if not np.isnan(mse_ext) else 0.0
    loss = alpha * loss_mid + (1 - alpha) * loss_ext
    print(f"hl={hl}天 → mse_mid={mse_mid:.5f}, mse_ext={mse_ext:.5f}, loss={loss:.5f}")
    results.append((hl, mse_mid, mse_ext, loss))
# 保存调优结果
df_tune = pd.DataFrame(results, columns=['half_life','mse_mid','mse_ext','loss'])
df_tune['best'] = df_tune['loss'] == df_tune['loss'].min()
df_tune.to_csv(TUNE_CSV, index=False)
best_hl = int(df_tune.loc[df_tune['best'], 'half_life'].iloc[0])
print("最佳半衰期:", best_hl)

# ─── 4. 生成日级健康度表 ─────────────────────────────────────────────
k = math.log(2) / best_hl
records = []
for sid in stations:
    sub = events_df[events_df.station_id == sid]
    dates_arr = sub['slice_date'].values
    li_arr    = sub['li'].values
    for D in dates:
        if sub.empty:
            s = 0.0
        else:
            diffs = np.abs((dates_arr - D).astype('timedelta64[D]').astype(int))
            w     = np.exp(-k * diffs)
            s     = (w * li_arr).sum() / w.sum()
        # 映射至 [0,1]
        health_conf = (s + 1.0) / (g_base + 1.0)
        records.append((sid, D, health_conf))
# 输出日级健康度
pd.DataFrame(records, columns=['station_id','date','health_conf'])\
    .to_csv(HEALTH_CSV, index=False)
print("已输出日级健康度表到", HEALTH_CSV)
