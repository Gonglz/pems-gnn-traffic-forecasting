#!/usr/bin/env python3
# coding: utf-8
"""
step40_fill_gpu_demo.py - GPU note(note)

note:
 1. noteoutput(step31_fillExter.csv), note(step30_logic_mask_continuous.csv),
    MD note(step32_md.csv), note(day_health_factor.csv)
 2. note(mask_flag), MD note(mask_md), note(mask_hf)
 3. noterowsnote: Local (GPU), Global (GPU), Temporal (GPU)
 4. outputnotetrainingnote(step40_interpolated.csv)

note:
  pip install pandas numpy scipy scikit-learn tqdm numba cupy cudf

note:
  cd finalproject/data_process
  python step40_fill_gpu_demo.py --test-n 100
  (traffic-env) lgong1@microway:/scratch/lgong1/finalproject/data_process$ python step40_fill_gpu_demo.py --test-n 100
► CPU readfirst 100 rowsnote
Loaded 100 rows, masks applied
Local (GPU) interpolation...
/scratch/lgong1/envs/traffic-env/lib/python3.10/site-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Global (GPU) interpolation...
Temporal (GPU) interpolation...
Temporal flow (GPU): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 34.13it/s]
Temporal occupancy (GPU): 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 1807.83it/s]
Temporal speed (GPU): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 1789.62it/s]
Step40 fill done, saved to /scratch/lgong1/finalproject/pems_data/step40_interpolated.csv
(traffic-env) lgong1@microway:/scratch/lgong1/finalproject/data_process$

"""

import os
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from scipy.interpolate import interp1d
from tqdm import tqdm
from numba import cuda
from sklearn.neighbors import KDTree
import argparse

# pathconfiguration
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_LONG_CSV = os.path.join(BASE_DIR, 'pems_data', 'step31_fillExter.csv')
LOGIC_CSV    = os.path.join(BASE_DIR, 'pems_data', 'step30_logic_mask_continuous.csv')
MD_CSV       = os.path.join(BASE_DIR, 'pems_data', 'step32_md.csv')
HF_CSV       = os.path.join(BASE_DIR, 'pems_data', 'day_health_factor.csv')
OUT_CSV      = os.path.join(BASE_DIR, 'pems_data', 'step40_interpolated.csv')

HF_THRESH   = 0.5
LOCAL_K     = 5
TEMP_METHOD = 'linear'

@cuda.jit
def local_kernel(feat_in, neighbor_idx, mask, feat_out, K):
    i = cuda.grid(1)
    if i < feat_in.size:
        if mask[i]:
            s = 0.0
            cnt = 0
            for j in range(K):
                idx = neighbor_idx[i, j]
                s += feat_in[idx]
                cnt += 1
            feat_out[i] = s / cnt if cnt > 0 else feat_in[i]
        else:
            feat_out[i] = feat_in[i]

def load_and_merge(test_n):
    # -- note: note pandas note CPU notereadfirst test_n rowsnotedata
    print(f"► CPU readfirst {test_n} rowsnote")
    pdf = pd.read_csv(RAW_LONG_CSV, nrows=test_n, parse_dates=['timestamp'])
    df = cudf.from_pandas(pdf)

    # -- note: readnote(CPU notefirst test_n rows)
    logic_pdf = pd.read_csv(LOGIC_CSV, nrows=test_n, parse_dates=['timestamp'])
    logic_pdf = logic_pdf[['timestamp','station_id','mask_logic']].rename(
        columns={'mask_logic':'mask_flag'}
    )
    logic_cudf = cudf.from_pandas(logic_pdf)
    df = df.merge(logic_cudf, on=['timestamp','station_id'], how='left')
    df['mask_flag'] = df['mask_flag'].fillna(0).astype('bool')

    # -- note: readnote MD note(CPU notefirst test_n rows)
    md_pdf = pd.read_csv(MD_CSV, nrows=test_n)
    md_cudf = cudf.from_pandas(md_pdf[['mask_md']])
    df['mask_md'] = md_cudf['mask_md'].astype('bool')

    # -- note: readnote(notefirst test_n rows)
    hf_pdf = pd.read_csv(HF_CSV, nrows=test_n, parse_dates=['date'])
    if 'health_factor' not in hf_pdf.columns:
        hf_pdf.rename(columns={hf_pdf.columns[-1]:'health_factor'}, inplace=True)
    hf_pdf['date'] = hf_pdf['date'].dt.date

    # note pdf note date note merge
    pdf['date'] = pdf['timestamp'].dt.date
    hf_merge = pdf[['station_id','date']].merge(
        hf_pdf[['station_id','date','health_factor']],
        on=['station_id','date'], how='left'
    )
    hf_cudf = cudf.from_pandas(hf_merge[['health_factor']])
    df['health_factor'] = hf_cudf['health_factor']
    df['mask_hf'] = df['health_factor'].fillna(1.0) < HF_THRESH

    # -- generatenote
    df['mask'] = df['mask_flag'] | df['mask_md'] | df['mask_hf']
    return df

def local_interpolate_gpu(df, feats):
    coords = df[['latitude','longitude']].to_pandas().values
    mask_coord = np.isfinite(coords).all(axis=1)
    mask_np = df['mask'].to_pandas().values
    bad_idx = np.where(mask_np & mask_coord)[0]
    good_idx = np.where(mask_coord)[0]
    tree = KDTree(coords[good_idx])
    nbrs = tree.query(coords[bad_idx], k=LOCAL_K+1, return_distance=False)[:,1:]

    N = len(df)
    out = {}
    for feat in feats:
        feat_p = df[feat].fillna(0).to_pandas().values.astype(np.float32)
        feat_in    = cp.asarray(feat_p)
        mask_gpu   = cp.asarray(mask_np, dtype=cp.bool_)
        neighbor_idx = cp.zeros((N, LOCAL_K), dtype=cp.int32)
        for pos, idx in enumerate(bad_idx):
            neighbor_idx[idx] = cp.asarray(good_idx[nbrs[pos]], dtype=cp.int32)
        feat_out = cp.empty_like(feat_in)
        threads = 256
        blocks  = (N + threads - 1) // threads
        local_kernel[blocks, threads](feat_in, neighbor_idx, mask_gpu, feat_out, LOCAL_K)
        out[feat] = cp.asnumpy(feat_out)
    return out

def global_interpolate_gpu(df, feats):
    cudf_df = df[['station_id','direction','mask'] + feats]
    normal  = cudf_df[~cudf_df['mask']]
    means   = normal.groupby(['station_id','direction'])[feats].mean().reset_index()
    merged  = df.merge(means, on=['station_id','direction'], how='left',
                       suffixes=('','_gmean')).to_pandas()
    mask_np = df['mask'].to_pandas().values
    out = {}
    for feat in feats:
        gm   = merged[f'{feat}_gmean'].values
        orig = df[feat].to_pandas().values
        out[feat] = np.where(mask_np, gm, orig)
    return out

def temporal_interpolate_gpu(df, feats):
    pdf      = df.to_pandas()
    ts_all   = cp.asarray(pdf['timestamp'].astype(np.int64).values)
    mask_all = cp.asarray(df['mask'].to_pandas().values, dtype=cp.bool_)
    groups   = pdf.groupby(['station_id','direction']).groups

    out = {}
    for feat in feats:
        vals_all = cp.asarray(pdf[feat].values.astype(np.float32))
        new_vals = vals_all.copy()
        for _, idx in tqdm(groups.items(), desc=f"Temporal {feat} (GPU)"):
            idx = np.array(idx, dtype=np.int32)
            ts   = ts_all[idx]
            vals = vals_all[idx]
            mask = mask_all[idx]
            if mask.all() or (~mask).all():
                continue
            xp = ts[~mask]; fp = vals[~mask]
            interp_vals = cp.interp(ts, xp, fp)
            new_vals[idx] = cp.where(mask, interp_vals, vals)
        out[feat] = cp.asnumpy(new_vals)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=0,
                        help='note: first N rows')
    args = parser.parse_args()

    df = load_and_merge(test_n=args.test_n)
    print(f"Loaded {len(df)} rows, masks applied")

    feats = ['flow','occupancy','speed']

    print('Local (GPU) interpolation...')
    local_map = local_interpolate_gpu(df, feats)
    for feat in feats:
        df[feat] = local_map[feat]

    print('Global (GPU) interpolation...')
    global_map = global_interpolate_gpu(df, feats)
    for feat in feats:
        df[feat] = global_map[feat]

    print('Temporal (GPU) interpolation...')
    temp_map = temporal_interpolate_gpu(df, feats)
    for feat in feats:
        df[feat] = temp_map[feat]

    # output
    df.to_pandas().to_csv(OUT_CSV, index=False)
    print('Step40 fill done, saved to', OUT_CSV)

if __name__ == '__main__':
    main()
