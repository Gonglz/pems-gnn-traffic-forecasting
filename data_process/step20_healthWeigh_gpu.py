#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 notegeneratenote(GPU note)

note:
 1. readnotefile, note:
    - Good note li = +g_base
    - Bad  note li = -1.0
 2. notexnote, computenote s(D)
 3. note"note"note"note"notecompute MSE, note alpha note
 4. note, note
 5. notecomputenote s(D), note [0,1] output day_health_factor.csv
 6. outputnoteresult step2_decay_tuning.csv

note CuPy note GPU note.
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step20_healthWeigh_gpu.py
GPU note: threshold=0.6, alpha=0.8
hl=0.1 note -> mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.2 note -> mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.3 note -> mse_mid=0.00250, mse_ext=0.00000, loss=0.00200
hl=0.5 note -> mse_mid=0.13566, mse_ext=0.00001, loss=0.10853
hl=0.7 note -> mse_mid=0.12731, mse_ext=0.00034, loss=0.10192
hl=1.0 note -> mse_mid=0.12859, mse_ext=0.00336, loss=0.10354
hl=5.0 note -> mse_mid=0.10552, mse_ext=0.15812, loss=0.11604
hl=7.0 note -> mse_mid=0.09231, mse_ext=0.24813, loss=0.12347
note: 0.3
noteoutputnote /scratch/lgong1/finalproject/pems_data/step20_day_health_factor_GPU.csv

process finished, exit codenote 0
"""

import pandas as pd
import numpy as np
import math
import cupy as cp
from pathlib import Path

# ─── configuration ─────────────────────────────────────────────────────────────
PEMS_DIR       = Path(__file__).resolve().parent.parent / 'pems_data'
DET_DIR        = PEMS_DIR / 'pems_detector'
RAW_LONG       = PEMS_DIR / 'step1_raw_long.csv'
TUNE_CSV       = PEMS_DIR / 'step2_decay_tuning.csv'
HEALTH_CSV     = PEMS_DIR / 'step20_day_health_factor_GPU.csv'
# note zero note, note
half_life_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 5.0, 7.0]
g_base         = 0.9       # Good note
threshold      = 0.6       # note
alpha          = 0.8       # note

# ─── 1. note ────────────────────────────────────────────────────
stations = pd.read_csv(RAW_LONG, usecols=['station_id'])['station_id'].astype(str).unique()
slice_dates, good_map = [], {}
for f in sorted(DET_DIR.glob('Detector Health *.xlsx')):
    date = pd.to_datetime(f.stem.split()[-1], format='%m%d%Y').date()
    slice_dates.append(date)
    dfh = pd.read_excel(f, sheet_name='Report Data')
    good_map[date] = set(dfh['VDS'].astype(str).str.strip())
# note: station_id, slice_date_int, li
events = []
for date in slice_dates:
    day_int = np.datetime64(date, 'D').astype(int)
    for sid in stations:
        li = g_base if sid in good_map[date] else -1.0
        events.append((sid, day_int, li))
events_df = pd.DataFrame(events, columns=['station_id','day_int','li'])
events_df.sort_values(['station_id','day_int'], inplace=True)
events_df.reset_index(drop=True, inplace=True)

# ─── 2. note GPU note ────────────────────────────────────────────
start_int = np.datetime64(min(slice_dates), 'D').astype(int)
end_int   = np.datetime64(max(slice_dates), 'D').astype(int)
days_range = np.arange(start_int, end_int+1)
# GPU note
days_gpu = cp.asarray(days_range)
# note station note
station_events = {sid: (grp['day_int'].values.astype(np.int32),
                        grp['li'].values.astype(np.float32))
                  for sid, grp in events_df.groupby('station_id', sort=False)}

# ─── 3. GPU note(note) ─────────────────────────────
results = []
print(f"GPU note: threshold={threshold}, alpha={alpha}")
for hl in half_life_list:
    k = math.log(2) / hl
    mse_mid_sum, mid_count = 0.0, 0
    ext_errors = []
    for sid, (days_arr, li_arr) in station_events.items():
        days_evt = cp.asarray(days_arr)
        li_evt   = cp.asarray(li_arr)
        diffs = cp.abs(days_evt[:, None] - days_gpu[None,:]).astype(cp.float32)
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
    print(f"hl={hl} note -> mse_mid={mse_mid:.5f}, mse_ext={mse_ext:.5f}, loss={loss:.5f}")
    results.append((hl, mse_mid, mse_ext, loss))
# savenoteresult
df_tune = pd.DataFrame(results, columns=['half_life','mse_mid','mse_ext','loss'])
df_tune['best'] = df_tune['loss']==df_tune['loss'].min()
df_tune.to_csv(TUNE_CSV, index=False)
# note best_hl note int, note
best_hl = df_tune.loc[df_tune['best'],'half_life'].iloc[0]
print(f"note: {best_hl}")

# ─── 4. GPU generatenote ──────────────────────────────────────────
k = math.log(2) / best_hl
records = []
for sid, (days_arr, li_arr) in station_events.items():
    days_evt = cp.asarray(days_arr)
    li_evt   = cp.asarray(li_arr)
    diffs = cp.abs(days_evt[:, None] - days_gpu[None,:]).astype(cp.float32)
    w     = cp.exp(-k * diffs)
    sum_w = w.sum(axis=0)
    s_all = (w * li_evt[:, None]).sum(axis=0) / sum_w
    s_cpu = cp.asnumpy(s_all)
    for i, d_int in enumerate(days_range):
        date_ts = pd.Timestamp(int(d_int), unit='D')
        health_conf = (s_cpu[i] + 1.0) / (g_base + 1.0)
        records.append((sid, date_ts, health_conf))
# outputnote
df_health = pd.DataFrame(records, columns=['station_id','date','health_conf'])
df_health.to_csv(HEALTH_CSV, index=False)
print("noteoutputnote", HEALTH_CSV)
