#!/usr/bin/env python3
# coding: utf-8
"""
step32_mdMask.py

note:
 1. notedata(flow, occupancy, speed, tavg, pcpn)note (is_weekend, is_holiday, in_custom_event)
 2. note, computenote μₖ note Σₖ(note), note Σₖ⁻¹
 3. note GPU noterowscomputenote Mahalanobis note
 4. note"note"detectionnote, generate mask_md
 5. outputnote md_squared, mask_md note step32_md.csv, notedistributionnote:
  pip install pandas numpy numba pyarrow matplotlib

note:
  cd finalproject/data_process
  python step32_MDmask.py

note:
 - note @cuda.jit note mahalanobis_kernel note CUDA kernel, noterowscomputenote.
 - resultnote:
     - step32_MDmask.csv: note, md_squared, mask_md
     - step32_thresholds.png: note
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step32_MDmask.py
step32_MDmask: notedatanote...
note 162169176 note, note 5
detectionnote 5 note GPU
note 0: 116842752 note(note): 35.4626
note 1: 45326424 note(note): 38.8974
note step32_md.csv...
note:../pems_data/step32_md.csv
✔ notesave:../pems_data/step32_thresholds.png

process finished, exit codenote 0

process finished, exit codenote 0
"""
import math
import numpy as np
import pandas as pd
from numba import cuda, float32

# note(note)
CONT_FEATURES = ['flow', 'occupancy', 'speed', 'tavg', 'pcpn']
D = len(CONT_FEATURES)

# ---------------------------
# CUDA Kernel
# note Numba @cuda.jit notefunctionnote CUDA kernel
# ---------------------------
@cuda.jit
def mahalanobis_kernel(X, mu, inv_cov, distances):
    i = cuda.grid(1)
    n = X.shape[0]
    if i < n:
        # note tmp note
        tmp = cuda.local.array(D, float32)
        for j in range(D):
            tmp[j] = X[i, j] - mu[j]
        # compute Σ⁻¹ * tmp
        y = cuda.local.array(D, float32)
        for j in range(D):
            acc = 0.0
            for k in range(D):
                acc += inv_cov[j, k] * tmp[k]
            y[j] = acc
        # compute Mahalanobis note
        dist2 = 0.0
        for j in range(D):
            dist2 += tmp[j] * y[j]
        distances[i] = dist2


def main():
    print('step32_MDmask: notedatanote...')
    df = pd.read_csv(
        '../pems_data/step31_fillExter.csv',
        usecols=CONT_FEATURES
                + ['is_weekend', 'is_holiday', 'in_custom_event']
                + ['station_id', 'timestamp']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # note NaN note
    # note, noterows
    means = df[CONT_FEATURES].mean()
    df[CONT_FEATURES] = df[CONT_FEATURES].fillna(means)

    # note: holiday*4 + event*2 + weekend*1
    grp = (df['is_holiday'].astype(int) * 4 +
           df['in_custom_event'].astype(int) * 2 +
           df['is_weekend'].astype(int))
    df['group'] = grp
    N = len(df)
    print(f'note {N} note, note {D}')

    # noteresultnote
    md_sq = np.zeros(N, dtype=np.float32)
    mask_md = np.zeros(N, dtype=bool)
    thresholds = {}

    # detectionnote GPU
    devices = list(cuda.gpus)
    print(f'detectionnote {len(devices)} note GPU')

    # noterowscompute
    for label in sorted(df['group'].unique()):
        idx = np.where(df['group'] == label)[0]
        size = len(idx)
        print(f'note {label}: {size} note')
        if size < D + 1:
            print('  note, note')
            continue
        # notedata
        Xg = df.iloc[idx][CONT_FEATURES].values.astype(np.float32)
        # computenote NaN
        mu = np.nanmean(Xg, axis=0).astype(np.float32)
        inds = np.where(np.isnan(Xg))
        if inds[0].size > 0:
            Xg[inds] = np.take(mu, inds[1])
        # note
        cov = np.cov(Xg, rowvar=False).astype(np.float32)
        cov += np.eye(D, dtype=np.float32) * 1e-6
        inv_cov = np.linalg.inv(cov).astype(np.float32)
        dist_g = np.zeros(size, dtype=np.float32)

        # note GPU noterows
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

        # note(notedetection)
        vals = np.sort(dist_g)
        n_vals = len(vals)
        if n_vals > 1:
            d_min, d_max = vals[0], vals[-1]
            if d_max > d_min:
                u = np.linspace(0, 1, n_vals, dtype=np.float32)
                v = (vals - d_min) / (d_max - d_min)
                diff = u - v
                knee_idx = int(np.argmax(diff))
                thr = vals[knee_idx]
            else:
                thr = vals[0]
        else:
            thr = vals[0]
        thresholds[label] = thr
        print(f'  note(note): {thr:.4f}')

        # noteresult
        md_sq[idx] = dist_g
        mask_md[idx] = dist_g > thr

    # output results
    print('note step32_md.csv...')
    out = pd.DataFrame({
        'station_id': df['station_id'],
        'timestamp': df['timestamp'],
        'md_squared': md_sq,
        'mask_md': mask_md
    })
    out.to_csv('../pems_data/step32_md.csv', index=False)

    print('note:../pems_data/step32_md.csv')

    # note
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        groups = list(thresholds.keys())
        values = [thresholds[g] for g in groups]
        ax.bar(groups, values, align='center')
        ax.set_xticks(groups)
        ax.set_xticklabels(groups)
        ax.set_xlabel('Group Label')
        ax.set_ylabel('Threshold (Knee Point)')
        ax.set_title('Group-wise Adaptive Mahalanobis Thresholds')
        ax.grid(True)
        fig_path = '../pems_data/step32_thresholds.png'
        fig.savefig(fig_path, dpi=300)
        print(f'✔ notesave: {fig_path}')
    except Exception as e:
        print(f'⚠ notefailed: {e}')

if __name__ == '__main__':
    main()