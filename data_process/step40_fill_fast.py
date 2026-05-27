#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step40_fill_fast.py - note(GPU note + note + Merge MD note)

note:
  cd finalproject/data_process
  python step40_fill_fast.py [--test-n N]

note:
  pandas numpy cudf cupy scikit-learn tqdm numba

description:
 1. note CPU note KDTree, note
 2. noteread, note:
    • note(note, MD, note)
    • note GPU + CuPy note Local / Global / Temporal note
 3. outputnote step40_interpolated.csv
 /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step40_fill_fast.py
▶ CPU notecomputenote…
readnote 4883 note
NaN note:
 latitude     0
longitude    0
dtype: int64
Processing chunks: 325it [6:20:44, 70.29s/it]
PASS note, resultnotesavenote /scratch/lgong1/finalproject/pems_data/step40_interpolated.csv

process finished, exit codenote 0

"""

import os
import argparse
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from sklearn.neighbors import KDTree as SKKDTree
from tqdm import tqdm

# --- path & note ---
BASE       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pems_data'))
RAW_LONG   = os.path.join(BASE, 'step31_fillExter.csv')
LOGIC_CSV  = os.path.join(BASE, 'step30_logic_mask_continuous.csv')
MD_CSV     = os.path.join(BASE, 'step32_md.csv')
HF_CSV     = os.path.join(BASE, 'day_health_factor.csv')
OUT_CSV    = os.path.join(BASE, 'step40_interpolated.csv')

HF_THRESH  = 0.5
LOCAL_K    = 5
CHUNK_SIZE = 500_000

# --- note ---
# note
logic_full = pd.read_csv(LOGIC_CSV, parse_dates=['timestamp'])
logic_full = logic_full[['timestamp','station_id','mask_logic']]

# Mahalanobis note
md_full = pd.read_csv(MD_CSV, parse_dates=['timestamp'])
md_full = md_full[['timestamp','station_id','mask_md']]

# note
hf_pdf = pd.read_csv(HF_CSV, parse_dates=['date'])
if 'health_factor' not in hf_pdf.columns:
    hf_pdf = hf_pdf.rename(columns={hf_pdf.columns[-1]:'health_factor'})
hf_pdf['date'] = hf_pdf['date'].dt.date
hf_pdf = hf_pdf[['station_id','date','health_factor']]

# --- CUDA Kernel for Local note ---
from numba import cuda, float32

@cuda.jit
def local_kernel(feat_in, nbr_idx, mask, feat_out, K):
    i = cuda.grid(1)
    if i < feat_in.size:
        if mask[i]:
            s = float32(0.0)
            for j in range(K):
                s += feat_in[nbr_idx[i, j]]
            feat_out[i] = s / K
        else:
            feat_out[i] = feat_in[i]

def process_chunk(pdf, is_first, neighbor_map):
    # note
    pdf = pdf.merge(logic_full, on=['timestamp','station_id'], how='left')
    pdf['mask_logic'] = pdf['mask_logic'].fillna(False)

    # note Mahalanobis note
    pdf = pdf.merge(md_full, on=['timestamp','station_id'], how='left')
    pdf['mask_md'] = pdf['mask_md'].fillna(False)

    # note
    pdf['date'] = pdf['timestamp'].dt.date
    pdf = pdf.merge(hf_pdf, on=['station_id','date'], how='left')
    pdf['health_factor'] = pdf['health_factor'].fillna(1.0)
    pdf['mask_hf'] = pdf['health_factor'] < HF_THRESH
    pdf.drop(columns=['date'], inplace=True)

    # note
    pdf['mask'] = pdf['mask_logic'] | pdf['mask_md'] | pdf['mask_hf']
    mask_np = pdf['mask'].to_numpy()

    # -- GPU Local note --
    # 1) noteinputnote
    N = len(pdf)
    coords = pdf[['latitude','longitude']].to_numpy()
    # 2) noterowsnote
    # noteID -> note neighbor_map note
    sid_to_pos = {sid: pos for pos, sid in enumerate(neighbor_map['station_ids'])}
    nbr_idx = np.zeros((N, LOCAL_K), dtype=np.int32)

    for i, sid in enumerate(pdf['station_id']):
        pos = sid_to_pos.get(sid, None)
        if pos is not None:
            # note, note nbr_idx
            nbr_idx[i,:] = neighbor_map['neighbors'][pos]
        # else: notecomputenote, nbr_idx[i] note [0,0,0,0,0]

    # 3) note kernel
    local_out = {}
    for feat in ['flow','occupancy','speed']:
        arr = pdf[feat].fillna(0).to_numpy(dtype=np.float32)
        feat_in     = cp.asarray(arr)
        mask_gpu    = cp.asarray(mask_np, dtype=cp.bool_)
        nbr_gpu     = cp.asarray(nbr_idx)
        feat_out    = cp.empty_like(feat_in)
        threads  = 256
        blocks   = (N + threads - 1) // threads
        local_kernel[blocks, threads](feat_in, nbr_gpu, mask_gpu, feat_out, LOCAL_K)
        local_out[feat] = cp.asnumpy(feat_out)

    # -- GPU Global note via cuDF --
    cdf    = cudf.from_pandas(pdf[['station_id','direction','mask','flow','occupancy','speed']])
    normal = cdf[~cdf['mask']]
    means  = normal.groupby(['station_id','direction']).mean().reset_index().to_pandas()
    merged = pdf.merge(means, on=['station_id','direction'], how='left', suffixes=('','_g'))
    global_out = {
        feat: np.where(pdf['mask'],
                       merged[f'{feat}_g'].to_numpy(),
                       pdf[feat].to_numpy())
        for feat in ['flow','occupancy','speed']
    }

    # -- GPU Temporal note via CuPy interp --
    temporal_out = {}
    ts_arr = cp.asarray(pdf['timestamp'].astype(np.int64).to_numpy())
    for feat in ['flow','occupancy','speed']:
        vals = cp.asarray(pdf[feat].to_numpy(dtype=np.float32))
        outv = vals.copy()
        for (_, grp) in pdf.groupby(['station_id','direction']).groups.items():
            idx = np.array(grp, dtype=np.int32)
            t = ts_arr[idx]; v = vals[idx]; m = cp.asarray(mask_np[idx])
            if m.all() or (~m).all():
                continue
            xp = t[~m]; fp = v[~m]
            iv = cp.interp(t, xp, fp)
            outv[idx] = cp.where(m, iv, v)
        temporal_out[feat] = cp.asnumpy(outv)

    # note: Local -> Global -> Temporal
    for feat in ['flow','occupancy','speed']:
        pdf[feat] = np.where(pdf['mask'], local_out[feat], pdf[feat])
        pdf[feat] = np.where(pdf['mask'], global_out[feat], pdf[feat])
        pdf[feat] = np.where(pdf['mask'], temporal_out[feat], pdf[feat])

    # note
    pdf.to_csv(OUT_CSV,
               index=False,
               mode='w' if is_first else 'a',
               header=is_first)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=0, help='note: first N rows')
    args = parser.parse_args()

    # --- CPU notecomputenote ---
    print("▶ CPU notecomputenote…")
    meta = pd.read_csv(RAW_LONG, usecols=['station_id','latitude','longitude'])
    station_meta = (meta.drop_duplicates('station_id').dropna(subset=['latitude','longitude']).reset_index(drop=True))
    print(f"readnote {len(station_meta)} note")
    print("NaN note: \n", station_meta[['latitude','longitude']].isnull().sum())

    coords = station_meta[['latitude','longitude']].to_numpy(dtype=np.float32)
    assert not np.isnan(coords).any(), "coords note NaN!"
    sk = SKKDTree(coords)
    nbr = sk.query(coords, k=LOCAL_K+1, return_distance=False)[:,1:]

    # note station_id noteneighbor position note dict
    neighbor_map = {
      'station_ids': station_meta['station_id'].to_numpy(),
      'neighbors': nbr
    }

    # --- note test note---
    first = True
    if args.test_n > 0:
        pdf = pd.read_csv(RAW_LONG, nrows=args.test_n, parse_dates=['timestamp'])
        process_chunk(pdf, True, neighbor_map)
    else:
        reader = pd.read_csv(RAW_LONG,
                             chunksize=CHUNK_SIZE,
                             parse_dates=['timestamp'])
        for pdf in tqdm(reader, desc='Processing chunks'):
            process_chunk(pdf, first, neighbor_map)
            first = False

    print("PASS note, resultnotesavenote", OUT_CSV)

if __name__ == '__main__':
    main()
