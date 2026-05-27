#!/usr/bin/env python3
# coding: utf-8
"""
step40_fill_gpu.py - note(GPU note + note + Merge MD note)

note:
  cd finalproject/data_process
  python step40_fill_gpu.py [--test-n N]

note:
  pandas numpy scikit-learn tqdm numba cupy cudf
Temporal speed (GPU):  92%|█████████▏| 4479/4888 [00:02<00:00, 2131.54it/s]
Temporal speed (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2117.53it/s]
PASS note, resultnotesavenote /scratch/lgong1/finalproject/pems_data/step40_interpolated.csv
Processing chunks: 325it [12:43:08, 140.89s/it]

process finished, exit codenote 0
process finished, exit codenote 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from numba import cuda
from sklearn.neighbors import KDTree
from tqdm import tqdm

# --- path & note ---
BASE       = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','pems_data'))
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

# Mahalanobis note(note station_id, timestamp)
md_full = pd.read_csv(MD_CSV, parse_dates=['timestamp'])
md_full = md_full[['timestamp','station_id','mask_md']]

# note
hf_pdf = pd.read_csv(HF_CSV, parse_dates=['date'])
if 'health_factor' not in hf_pdf.columns:
    hf_pdf = hf_pdf.rename(columns={hf_pdf.columns[-1]:'health_factor'})
hf_pdf['date'] = hf_pdf['date'].dt.date
hf_pdf = hf_pdf[['station_id','date','health_factor']]

# --- CUDA Kernel for Local note ---
@cuda.jit
def local_kernel(feat_in, nbr_idx, mask, feat_out, K):
    i = cuda.grid(1)
    if i < feat_in.size:
        if mask[i]:
            s = 0.0
            cnt = 0
            for j in range(K):
                idx = nbr_idx[i, j]
                if idx >= 0:
                    s += feat_in[idx]
                    cnt += 1
            feat_out[i] = s / cnt if cnt>0 else feat_in[i]
        else:
            feat_out[i] = feat_in[i]

def process_chunk(pdf: pd.DataFrame, is_first: bool):
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
    mask_np = pdf['mask'].to_numpy(dtype=bool)
    for feat in ['flow','occupancy','speed']:
        pdf.loc[mask_np, feat] = np.nan

    # -- GPU Local note --
    coords = pdf[['latitude','longitude']].values
    valid  = np.isfinite(coords).all(axis=1)
    bad_idx  = np.where(mask_np & valid)[0]
    good_idx = np.where((~mask_np) & valid)[0]
    nbr_idx = np.full((len(pdf), LOCAL_K), -1, dtype=np.int32)
    if len(bad_idx) > 0 and len(good_idx) > 0:
        tree = KDTree(coords[good_idx])
        k_local = min(LOCAL_K, len(good_idx))
        nbrs = tree.query(coords[bad_idx], k=k_local, return_distance=False)
        nbr_idx[bad_idx,:k_local] = good_idx[nbrs]

    N = len(pdf)
    local_out = {}
    for feat in ['flow','occupancy','speed']:
        arr = pdf[feat].fillna(0).astype(np.float32).values
        feat_in     = cp.asarray(arr)
        mask_gpu    = cp.asarray(mask_np, dtype=cp.bool_)
        nbr_idx_gpu = cp.asarray(nbr_idx)
        feat_out = cp.empty_like(feat_in)
        threads  = 256
        blocks   = (N + threads - 1)//threads
        local_kernel[blocks, threads](feat_in, nbr_idx_gpu, mask_gpu, feat_out, LOCAL_K)
        local_out[feat] = cp.asnumpy(feat_out)

    # -- GPU Global note via cuDF --
    cdf    = cudf.from_pandas(pdf[['station_id','direction','mask','flow','occupancy','speed']])
    normal = cdf[~cdf['mask']]
    means  = normal.groupby(['station_id','direction']).mean().reset_index().to_pandas()
    merged = pdf.merge(means, on=['station_id','direction'], how='left', suffixes=('','_gmean'))
    global_out = {}
    for feat in ['flow','occupancy','speed']:
        gm   = merged[f'{feat}_gmean'].values
        orig = pdf[feat].values
        global_out[feat] = np.where(mask_np, gm, orig)

    # -- GPU Temporal note via CuPy interp --
    tpdf   = pdf
    ts_all = cp.asarray(tpdf['timestamp'].astype(np.int64).values)
    mask_all = cp.asarray(mask_np, dtype=cp.bool_)
    groups   = tpdf.groupby(['station_id','direction']).groups

    temporal_out = {}
    for feat in ['flow','occupancy','speed']:
        vals_all = cp.asarray(tpdf[feat].values.astype(np.float32))
        out_vals = vals_all.copy()
        for _, idx in tqdm(groups.items(), desc=f"Temporal {feat} (GPU)"):
            idx_arr = np.array(idx, dtype=np.int32)
            ts = ts_all[idx_arr]; vs = vals_all[idx_arr]; mk = mask_all[idx_arr]
            if mk.all() or (~mk).all(): continue
            xp = ts[~mk]; fp = vs[~mk]
            interp_vals = cp.interp(ts, xp, fp)
            out_vals[idx_arr] = cp.where(mk, interp_vals, vs)
        temporal_out[feat] = cp.asnumpy(out_vals)

    # noteresult
    for feat in ['flow','occupancy','speed']:
        filled = pdf[feat].to_numpy(dtype=np.float32, copy=True)
        for candidate in (local_out[feat], global_out[feat], temporal_out[feat]):
            candidate = np.asarray(candidate, dtype=np.float32)
            needs_fill = mask_np & ~np.isfinite(filled) & np.isfinite(candidate)
            filled = np.where(needs_fill, candidate, filled)
        pdf[feat] = filled

    # note CSV
    pdf.to_csv(OUT_CSV,
               index=False,
               mode='w' if is_first else 'a',
               header=is_first)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=0, help='note: first N rows')
    args = parser.parse_args()

    first = True
    if args.test_n > 0:
        pdf = pd.read_csv(RAW_LONG, nrows=args.test_n, parse_dates=['timestamp'])
        process_chunk(pdf, is_first=True)
    else:
        reader = pd.read_csv(RAW_LONG,
                             chunksize=CHUNK_SIZE,
                             parse_dates=['timestamp'])
        for pdf in tqdm(reader, desc='Processing chunks'):
            process_chunk(pdf, is_first=first)
            first = False

    print("PASS note, resultnotesavenote", OUT_CSV)

if __name__ == '__main__':
    main()
""""Processing chunks: 6it [19:22, 175.76s/it]
Temporal flow (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal flow (GPU):   4%|▍         | 205/4888 [00:00<00:02, 2046.76it/s]
Temporal flow (GPU):   8%|▊         | 412/4888 [00:00<00:02, 2058.62it/s]
Temporal flow (GPU):  13%|█▎        | 618/4888 [00:00<00:02, 2041.26it/s]
Temporal flow (GPU):  17%|█▋        | 823/4888 [00:00<00:02, 2005.84it/s]
Temporal flow (GPU):  21%|██        | 1034/4888 [00:00<00:01, 2041.89it/s]
Temporal flow (GPU):  25%|██▌       | 1244/4888 [00:00<00:01, 2059.23it/s]
Temporal flow (GPU):  30%|██▉       | 1456/4888 [00:00<00:01, 2075.98it/s]
Temporal flow (GPU):  34%|███▍      | 1666/4888 [00:00<00:01, 2080.96it/s]
Temporal flow (GPU):  38%|███▊      | 1875/4888 [00:00<00:01, 2061.13it/s]
Temporal flow (GPU):  43%|████▎     | 2082/4888 [00:01<00:01, 2061.95it/s]
Temporal flow (GPU):  47%|████▋     | 2294/4888 [00:01<00:01, 2078.55it/s]
Temporal flow (GPU):  51%|█████     | 2502/4888 [00:01<00:01, 2077.65it/s]
Temporal flow (GPU):  56%|█████▌    | 2715/4888 [00:01<00:01, 2091.19it/s]
Temporal flow (GPU):  60%|█████▉    | 2925/4888 [00:01<00:00, 2082.05it/s]
Temporal flow (GPU):  64%|██████▍   | 3136/4888 [00:01<00:00, 2090.14it/s]
Temporal flow (GPU):  68%|██████▊   | 3346/4888 [00:01<00:00, 2084.33it/s]
Temporal flow (GPU):  73%|███████▎  | 3555/4888 [00:01<00:00, 2078.63it/s]
Temporal flow (GPU):  77%|███████▋  | 3763/4888 [00:01<00:00, 2077.28it/s]
Temporal flow (GPU):  81%|████████  | 3971/4888 [00:01<00:00, 2043.52it/s]
Temporal flow (GPU):  85%|████████▌ | 4176/4888 [00:02<00:00, 2045.10it/s]
Temporal flow (GPU):  90%|████████▉ | 4387/4888 [00:02<00:00, 2062.64it/s]
Temporal flow (GPU):  94%|█████████▍| 4594/4888 [00:02<00:00, 2063.27it/s]
Temporal flow (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2066.15it/s]

Temporal occupancy (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal occupancy (GPU):   4%|▍         | 203/4888 [00:00<00:02, 2027.43it/s]
Temporal occupancy (GPU):   8%|▊         | 407/4888 [00:00<00:02, 2032.36it/s]
Temporal occupancy (GPU):  12%|█▎        | 611/4888 [00:00<00:02, 2030.96it/s]
Temporal occupancy (GPU):  17%|█▋        | 815/4888 [00:00<00:02, 1984.45it/s]
Temporal occupancy (GPU):  21%|██        | 1023/4888 [00:00<00:01, 2017.71it/s]
Temporal occupancy (GPU):  25%|██▌       | 1231/4888 [00:00<00:01, 2035.91it/s]
Temporal occupancy (GPU):  30%|██▉       | 1442/4888 [00:00<00:01, 2058.17it/s]
Temporal occupancy (GPU):  34%|███▍      | 1650/4888 [00:00<00:01, 2064.08it/s]
Temporal occupancy (GPU):  38%|███▊      | 1857/4888 [00:00<00:01, 2054.65it/s]
Temporal occupancy (GPU):  42%|████▏     | 2066/4888 [00:01<00:01, 2062.93it/s]
Temporal occupancy (GPU):  47%|████▋     | 2273/4888 [00:01<00:01, 2057.64it/s]
Temporal occupancy (GPU):  51%|█████     | 2479/4888 [00:01<00:01, 2057.76it/s]
Temporal occupancy (GPU):  55%|█████▌    | 2692/4888 [00:01<00:01, 2077.82it/s]
Temporal occupancy (GPU):  59%|█████▉    | 2900/4888 [00:01<00:00, 2073.48it/s]
Temporal occupancy (GPU):  64%|██████▎   | 3108/4888 [00:01<00:00, 2060.36it/s]
Temporal occupancy (GPU):  68%|██████▊   | 3315/4888 [00:01<00:00, 2062.63it/s]
Temporal occupancy (GPU):  72%|███████▏  | 3522/4888 [00:01<00:00, 2056.45it/s]
Temporal occupancy (GPU):  76%|███████▋  | 3728/4888 [00:01<00:00, 2052.91it/s]
Temporal occupancy (GPU):  80%|████████  | 3934/4888 [00:01<00:00, 2040.93it/s]
Temporal occupancy (GPU):  85%|████████▍ | 4139/4888 [00:02<00:00, 2030.88it/s]
Temporal occupancy (GPU):  89%|████████▉ | 4354/4888 [00:02<00:00, 2063.53it/s]
Temporal occupancy (GPU):  93%|█████████▎| 4561/4888 [00:02<00:00, 2055.54it/s]
Temporal occupancy (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2053.33it/s]

Temporal speed (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal speed (GPU):   4%|▍         | 203/4888 [00:00<00:02, 2027.72it/s]
Temporal speed (GPU):   8%|▊         | 406/4888 [00:00<00:02, 2019.43it/s]
Temporal speed (GPU):  12%|█▏        | 610/4888 [00:00<00:02, 2027.97it/s]
Temporal speed (GPU):  17%|█▋        | 813/4888 [00:00<00:02, 2008.88it/s]
Temporal speed (GPU):  21%|██        | 1022/4888 [00:00<00:01, 2034.63it/s]
Temporal speed (GPU):  25%|██▌       | 1232/4888 [00:00<00:01, 2053.60it/s]
Temporal speed (GPU):  30%|██▉       | 1444/4888 [00:00<00:01, 2072.59it/s]
Temporal speed (GPU):  34%|███▍      | 1653/4888 [00:00<00:01, 2077.46it/s]
Temporal speed (GPU):  38%|███▊      | 1861/4888 [00:00<00:01, 2066.42it/s]
Temporal speed (GPU):  42%|████▏     | 2070/4888 [00:01<00:01, 2071.98it/s]
Temporal speed (GPU):  47%|████▋     | 2281/4888 [00:01<00:01, 2081.04it/s]
Temporal speed (GPU):  51%|█████     | 2490/4888 [00:01<00:01, 2073.84it/s]
Temporal speed (GPU):  55%|█████▌    | 2705/4888 [00:01<00:01, 2093.90it/s]
Temporal speed (GPU):  60%|█████▉    | 2915/4888 [00:01<00:00, 2051.25it/s]
Temporal speed (GPU):  64%|██████▍   | 3127/4888 [00:01<00:00, 2069.20it/s]
Temporal speed (GPU):  68%|██████▊   | 3336/4888 [00:01<00:00, 2073.03it/s]
Temporal speed (GPU):  73%|███████▎  | 3544/4888 [00:01<00:00, 2069.70it/s]
Temporal speed (GPU):  77%|███████▋  | 3753/4888 [00:01<00:00, 2073.35it/s]
Temporal speed (GPU):  81%|████████  | 3961/4888 [00:01<00:00, 2052.70it/s]
Temporal speed (GPU):  85%|████████▌ | 4167/4888 [00:02<00:00, 2049.58it/s]
Temporal speed (GPU):  90%|████████▉ | 4378/4888 [00:02<00:00, 2066.96it/s]
Temporal speed (GPU):  94%|█████████▍| 4586/4888 [00:02<00:00, 2068.56it/s]
Temporal speed (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2064.59it/s]
Processing chunks: 7it [25:13, 233.18s/it]
Temporal flow (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal flow (GPU):   3%|▎         | 170/4888 [00:00<00:02, 1699.60it/s]
Temporal flow (GPU):   7%|▋         | 361/4888 [00:00<00:02, 1820.95it/s]
Temporal flow (GPU):  12%|█▏        | 575/4888 [00:00<00:02, 1964.28it/s]
Temporal flow (GPU):  16%|█▌        | 773/4888 [00:00<00:02, 1968.16it/s]
Temporal flow (GPU):  20%|██        | 987/4888 [00:00<00:01, 2028.68it/s]
Temporal flow (GPU):  25%|██▍       | 1199/4888 [00:00<00:01, 2058.87it/s]
Temporal flow (GPU):  29%|██▉       | 1416/4888 [00:00<00:01, 2092.22it/s]
Temporal flow (GPU):  33%|███▎      | 1630/4888 [00:00<00:01, 2105.93it/s]
Temporal flow (GPU):  38%|███▊      | 1841/4888 [00:00<00:01, 2104.00it/s]
Temporal flow (GPU):  42%|████▏     | 2055/4888 [00:01<00:01, 2112.99it/s]
Temporal flow (GPU):  46%|████▋     | 2271/4888 [00:01<00:01, 2127.12it/s]
Temporal flow (GPU):  51%|█████     | 2484/4888 [00:01<00:01, 2127.16it/s]
Temporal flow (GPU):  55%|█████▌    | 2703/4888 [00:01<00:01, 2145.67it/s]
Temporal flow (GPU):  60%|█████▉    | 2918/4888 [00:01<00:00, 2141.06it/s]
Temporal flow (GPU):  64%|██████▍   | 3134/4888 [00:01<00:00, 2145.69it/s]
Temporal flow (GPU):  69%|██████▊   | 3349/4888 [00:01<00:00, 2141.69it/s]
Temporal flow (GPU):  73%|███████▎  | 3564/4888 [00:01<00:00, 2128.84it/s]
Temporal flow (GPU):  77%|███████▋  | 3780/4888 [00:01<00:00, 2136.71it/s]
Temporal flow (GPU):  82%|████████▏ | 3994/4888 [00:01<00:00, 2114.39it/s]
Temporal flow (GPU):  86%|████████▌ | 4212/4888 [00:02<00:00, 2132.86it/s]
Temporal flow (GPU):  91%|█████████ | 4432/4888 [00:02<00:00, 2152.10it/s]
Temporal flow (GPU):  95%|█████████▌| 4648/4888 [00:02<00:00, 2144.53it/s]
Temporal flow (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2103.35it/s]

Temporal occupancy (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal occupancy (GPU):   4%|▍         | 209/4888 [00:00<00:02, 2086.31it/s]
Temporal occupancy (GPU):   9%|▊         | 421/4888 [00:00<00:02, 2101.37it/s]
Temporal occupancy (GPU):  13%|█▎        | 632/4888 [00:00<00:02, 2082.09it/s]
Temporal occupancy (GPU):  17%|█▋        | 841/4888 [00:00<00:01, 2070.18it/s]
Temporal occupancy (GPU):  22%|██▏       | 1056/4888 [00:00<00:01, 2096.43it/s]
Temporal occupancy (GPU):  26%|██▌       | 1271/4888 [00:00<00:01, 2112.96it/s]
Temporal occupancy (GPU):  30%|███       | 1483/4888 [00:00<00:01, 2113.93it/s]
Temporal occupancy (GPU):  35%|███▍      | 1697/4888 [00:00<00:01, 2119.47it/s]
Temporal occupancy (GPU):  39%|███▉      | 1909/4888 [00:00<00:01, 2108.51it/s]
Temporal occupancy (GPU):  43%|████▎     | 2125/4888 [00:01<00:01, 2123.35it/s]
Temporal occupancy (GPU):  48%|████▊     | 2339/4888 [00:01<00:01, 2126.00it/s]
Temporal occupancy (GPU):  52%|█████▏    | 2555/4888 [00:01<00:01, 2133.79it/s]
Temporal occupancy (GPU):  57%|█████▋    | 2771/4888 [00:01<00:00, 2139.57it/s]
Temporal occupancy (GPU):  61%|██████    | 2985/4888 [00:01<00:00, 2135.13it/s]
Temporal occupancy (GPU):  65%|██████▌   | 3201/4888 [00:01<00:00, 2140.92it/s]
Temporal occupancy (GPU):  70%|██████▉   | 3416/4888 [00:01<00:00, 2122.59it/s]
Temporal occupancy (GPU):  74%|███████▍  | 3629/4888 [00:01<00:00, 2124.57it/s]
Temporal occupancy (GPU):  79%|███████▊  | 3842/4888 [00:01<00:00, 2120.52it/s]
Temporal occupancy (GPU):  83%|████████▎ | 4055/4888 [00:01<00:00, 2109.28it/s]
Temporal occupancy (GPU):  87%|████████▋ | 4273/4888 [00:02<00:00, 2128.92it/s]
Temporal occupancy (GPU):  92%|█████████▏| 4488/4888 [00:02<00:00, 2132.55it/s]
Temporal occupancy (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2122.18it/s]

Temporal speed (GPU):   0%|          | 0/4888 [00:00<?,?it/s]
Temporal speed (GPU):   4%|▍         | 209/4888 [00:00<00:02, 2079.64it/s]
Temporal speed (GPU):   9%|▊         | 421/4888 [00:00<00:02, 2101.83it/s]
Temporal speed (GPU):  13%|█▎        | 632/4888 [00:00<00:02, 2080.97it/s]
Temporal speed (GPU):  17%|█▋        | 841/4888 [00:00<00:01, 2071.18it/s]
Temporal speed (GPU):  22%|██▏       | 1056/4888 [00:00<00:01, 2097.35it/s]
Temporal speed (GPU):  26%|██▌       | 1273/4888 [00:00<00:01, 2119.68it/s]
Temporal speed (GPU):  30%|███       | 1486/4888 [00:00<00:01, 2122.85it/s]
Temporal speed (GPU):  35%|███▍      | 1701/4888 [00:00<00:01, 2129.80it/s]
Temporal speed (GPU):  39%|███▉      | 1914/4888 [00:00<00:01, 2112.80it/s]
Temporal speed (GPU):  44%|████▎     | 2130/4888 [00:01<00:01, 2126.74it/s]
Temporal speed (GPU):  48%|████▊     | 2346/4888 [00:01<00:01, 2134.01it/s]
Temporal speed (GPU):  52%|█████▏    | 2561/4888 [00:01<00:01, 2138.72it/s]
Temporal speed (GPU):  57%|█████▋    | 2778/4888 [00:01<00:00, 2147.57it/s]
Temporal speed (GPU):  61%|██████    | 2993/4888 [00:01<00:00, 2139.43it/s]
Temporal speed (GPU):  66%|██████▌   | 3211/4888 [00:01<00:00, 2149.68it/s]
Temporal speed (GPU):  70%|███████   | 3426/4888 [00:01<00:00, 2129.41it/s]
Temporal speed (GPU):  74%|███████▍  | 3641/4888 [00:01<00:00, 2134.16it/s]
Temporal speed (GPU):  79%|███████▉  | 3855/4888 [00:01<00:00, 2128.09it/s]
Temporal speed (GPU):  83%|████████▎ | 4068/4888 [00:01<00:00, 2122.60it/s]
Temporal speed (GPU):  88%|████████▊ | 4285/4888 [00:02<00:00, 2135.74it/s]
Temporal speed (GPU):  92%|█████████▏| 4499/4888 [00:02<00:00, 2134.69it/s]
Temporal speed (GPU): 100%|██████████| 4888/4888 [00:02<00:00, 2127.92it/s]
Processing chunks: 8it [32:27, 297.18s/it]"""
