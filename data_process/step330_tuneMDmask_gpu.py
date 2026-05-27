#!/usr/bin/env python3
# coding: utf-8
"""
step31_md_cuda.py

note:
 1. notedata(flow, occupancy, speed, tavg, pcpn)note (is_weekend, is_holiday, in_custom_event)
 2. note, computenote μₖ note Σₖ(note), note Σₖ⁻¹
 3. note GPU noterowscomputenote Mahalanobis note
 4. note 95% note, generate mask_md
 5. outputnote md_squared note mask_md note step32_md.csv

note:
  pip install pandas numpy numba pyarrow

note:
  cd finalproject/data_process
  python step31_md_cuda.py
"""
import math
import numpy as np
import pandas as pd
from numba import cuda, float32

# note(note)
CONT_FEATURES = ['flow', 'occupancy', 'speed', 'tavg', 'pcpn']
D = len(CONT_FEATURES)

@cuda.jit
def mahalanobis_kernel(X, mu, inv_cov, distances):
    i = cuda.grid(1)
    n = X.shape[0]
    if i < n:
        tmp = cuda.local.array(D, float32)
        for j in range(D):
            tmp[j] = X[i, j] - mu[j]
        y = cuda.local.array(D, float32)
        for j in range(D):
            acc = 0.0
            for k in range(D):
                acc += inv_cov[j, k] * tmp[k]
            y[j] = acc
        dist2 = 0.0
        for j in range(D):
            dist2 += tmp[j] * y[j]
        distances[i] = dist2


def main():
    print('notedatanote...')
    df = pd.read_csv('../pems_data/step31_fillExter.csv', usecols=CONT_FEATURES + ['is_weekend', 'is_holiday', 'in_custom_event'])
    # note NaN note, notecomputenote
    df = df.dropna(subset=CONT_FEATURES).reset_index(drop=True)
    # note
    grp = (df['is_holiday'].astype(int) * 4 +
           df['in_custom_event'].astype(int) * 2 +
           df['is_weekend'].astype(int))
    df['group'] = grp
    N = len(df)
    print(f'note {N} note, note {D}')

    # resultnote
    md_sq = np.zeros(N, dtype=np.float32)
    mask_md = np.zeros(N, dtype=bool)
    thresholds = {}  # note(N, dtype=bool)

    devices = list(cuda.gpus)
    print(f'detectionnote {len(devices)} note GPU')

    # noterowsnote
    for label in sorted(df['group'].unique()):
        idx = np.where(df['group'] == label)[0]
        size = len(idx)
        print(f'note {label}: {size} note')
        if size < D + 1:
            print('  note, note')
            continue
        Xg = df.iloc[idx][CONT_FEATURES].values.astype(np.float32)
        # notemissing valuesnoterowsnote, note NaN note NaN
        mu = np.nanmean(Xg, axis=0).astype(np.float32)
        # note NaN note
        inds = np.where(np.isnan(Xg))
        if inds[0].size > 0:
            Xg[inds] = np.take(mu, inds[1])
        # computenote, note
        cov = np.cov(Xg, rowvar=False).astype(np.float32)
        cov += np.eye(D, dtype=np.float32) * 1e-6
        inv_cov = np.linalg.inv(cov).astype(np.float32)
        dist_g = np.zeros(size, dtype=np.float32)

        # GPU noterows
        chunk = math.ceil(size / len(devices))
        for dev_id, dev in enumerate(devices):
            start = dev_id * chunk
            end = min((dev_id + 1) * chunk, size)
            if start >= end:
                break
            Xc = np.ascontiguousarray(Xg[start:end])
            with dev:
                d_X = cuda.to_device(Xc)
                d_mu = cuda.to_device(mu)
                d_ic = cuda.to_device(inv_cov)
                d_dist = cuda.device_array(end - start, dtype=np.float32)
                threads = 256
                blocks = math.ceil((end - start) / threads)
                mahalanobis_kernel[blocks, threads](d_X, d_mu, d_ic, d_dist)
                dist_g[start:end] = d_dist.copy_to_host()

        thr = np.quantile(dist_g, 0.95)
        thresholds[label] = thr  # note
        print(f'  note(95%): {thr:.4f}')
        md_sq[idx] = dist_g
        mask_md[idx] = dist_g > thr

    print('noteresult...')
    out = df.copy()
    out['md_squared'] = md_sq
    out['mask_md'] = mask_md
    out.to_csv('../pems_data/step32_md.csv', index=False)
    print('note:../pems_data/step32_md.csv')

    # note
    try:
        import matplotlib.pyplot as plt
        groups = list(thresholds.keys())
        values = [thresholds[g] for g in groups]
        plt.figure()
        plt.bar(groups, values)
        plt.xlabel('Group Label')
        plt.ylabel('Mahalanobis Threshold (95%)')
        plt.title('Group-wise Mahalanobis Thresholds')
        plt.grid(True)
        fig_path = '../pems_data/step33_thresholds.png'
        plt.savefig(fig_path, dpi=300)
        print(f'✔ notesave: {fig_path}')
    except Exception as e:
        print(f'⚠ notefailed: {e}')

if __name__ == '__main__':
    main()
