#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 notegeneratenote(note: note+note)

note:
 1. readnotefile, note:
    - Good note li = +g_base
    - Bad  note li = -1.0
 2. notexnote, computenote s(D)
 3. note"note"note"note"notecompute MSE, note alpha note
 4. note, note
 5. notecomputenote s(D), note [0,1] output day_health_factor.csv
 6. outputnoteresult step2_decay_tuning.csv

 /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step200_healthWeight.py
note, threshold=0.60, alpha=0.80
hl=1note -> mse_mid=0.12859, mse_ext=0.00336, loss=0.10354
hl=2note -> mse_mid=0.16528, mse_ext=0.03900, loss=0.14002
hl=3note -> mse_mid=0.12071, mse_ext=0.08043, loss=0.11265
hl=5note -> mse_mid=0.10671, mse_ext=0.15812, loss=0.11699
hl=7note -> mse_mid=0.08838, mse_ext=0.24813, loss=0.12033
hl=10note -> mse_mid=0.05564, mse_ext=0.37557, loss=0.11962
hl=14note -> mse_mid=0.04474, mse_ext=0.49858, loss=0.13551
hl=21note -> mse_mid=0.03328, mse_ext=0.62205, loss=0.15104
note: 1
noteoutputnote /scratch/lgong1/finalproject/pems_data/day_health_factor.csv
note
GPU note float32 note, CPU note float64(NumPy default).note, GPU note w = cp.exp(-k*diffs.astype(cp.float64)).astype(cp.float64), note float64 note.
process finished, exit codenote 0


output:
 - pems_data/step2_decay_tuning.csv   (half_life, mse_mid, mse_ext, loss, best)
 - pems_data/day_health_factor.csv    (station_id, date, health_conf)
"""
import pandas as pd
import numpy as np
import math
from pathlib import Path

# ─── configuration ─────────────────────────────────────────────────────────────
PEMS_DIR       = Path(__file__).resolve().parent.parent / 'pems_data'
DET_DIR        = PEMS_DIR / 'pems_detector'
RAW_LONG       = PEMS_DIR / 'step1_raw_long.csv'
TUNE_CSV       = PEMS_DIR / 'step2_decay_tuning.csv'
HEALTH_CSV     = PEMS_DIR / 'day_health_factor.csv'
half_life_list = [1, 2, 3, 5, 7, 10, 14, 21]
g_base         = 0.9       # Good note
threshold      = 0.6       # note
alpha          = 0.8       # note

# ─── 1. note ────────────────────────────────────────────────────
# readnote
stations = pd.read_csv(RAW_LONG, usecols=['station_id'])['station_id'].astype(str).unique()
# note Good note
slice_dates = []
good_map = {}
for f in sorted(DET_DIR.glob('Detector Health *.xlsx')):
    date = pd.to_datetime(f.stem.split()[-1], format='%m%d%Y').date()
    slice_dates.append(date)
    dfh = pd.read_excel(f, sheet_name='Report Data')
    good_map[date] = set(dfh['VDS'].astype(str).str.strip())
# generatenote: station_id, slice_date, li
events = []
for date in slice_dates:
    goods = good_map[date]
    for sid in stations:
        li = g_base if sid in goods else -1.0
        events.append((sid, date, li))
events_df = pd.DataFrame(events, columns=['station_id','slice_date','li'])
events_df.sort_values(['station_id','slice_date'], inplace=True)
events_df.reset_index(drop=True, inplace=True)

# ─── 2. note ─────────────────────────────────────────────────────
start_date = min(slice_dates)
end_date   = max(slice_dates)
dates = pd.date_range(start_date, end_date, freq='D').date
calendar = [(sid, d) for sid in stations for d in dates]

# ─── 3. note(note) ─────────────────────────────────────
results = []
print("note, threshold=%.2f, alpha=%.2f" % (threshold, alpha))
for hl in half_life_list:
    k = math.log(2) / hl
    errors_mid = []  # |s|<=threshold note 0 note
    errors_ext = []  # note li note
    # note
    for sid, D in calendar:
        sub = events_df[events_df.station_id == sid]
        if sub.empty:
            continue
        # compute s(D)
        dates_arr = sub['slice_date'].values
        li_arr    = sub['li'].values
        diffs = np.abs((dates_arr - D).astype('timedelta64[D]').astype(int))
        w     = np.exp(-k * diffs)
        s     = (w * li_arr).sum() / w.sum()
        # note
        if D in dates_arr:
            # note, note li
            # note, note
            mask    = dates_arr == np.datetime64(D)
            li_vals = li_arr[mask]
            true_li = float(li_vals[0]) if li_vals.size >= 1 else 0.0
            errors_ext.append((s - true_li) ** 2)
        elif abs(s) <= threshold:
            # note, note 0 note
            errors_mid.append(s ** 2)
        # note
    mse_mid = np.mean(errors_mid) if errors_mid else np.nan
    mse_ext = np.mean(errors_ext) if errors_ext else np.nan
    # note
    loss_mid = mse_mid if not np.isnan(mse_mid) else 0.0
    loss_ext = mse_ext if not np.isnan(mse_ext) else 0.0
    loss = alpha * loss_mid + (1 - alpha) * loss_ext
    print(f"hl={hl}note -> mse_mid={mse_mid:.5f}, mse_ext={mse_ext:.5f}, loss={loss:.5f}")
    results.append((hl, mse_mid, mse_ext, loss))
# savenoteresult
df_tune = pd.DataFrame(results, columns=['half_life','mse_mid','mse_ext','loss'])
df_tune['best'] = df_tune['loss'] == df_tune['loss'].min()
df_tune.to_csv(TUNE_CSV, index=False)
best_hl = int(df_tune.loc[df_tune['best'], 'half_life'].iloc[0])
print("note:", best_hl)

# ─── 4. generatenote ─────────────────────────────────────────────
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
        # note [0,1]
        health_conf = (s + 1.0) / (g_base + 1.0)
        records.append((sid, D, health_conf))
# outputnote
pd.DataFrame(records, columns=['station_id','date','health_conf']).to_csv(HEALTH_CSV, index=False)
print("noteoutputnote", HEALTH_CSV)
