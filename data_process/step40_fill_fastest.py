"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from numba import cuda, float32
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import dask_cudf
from dask.distributed import progress
from cuml.neighbors import NearestNeighbors

BASE_DIR = '/scratch/lgong1/finalproject/pems_data'
MASK_PARQUET = os.path.join(BASE_DIR, 'step34_maskMix.parquet')
STATIONS_CSV = os.path.join(BASE_DIR, 'step01_d07_meta.csv')
RAW_CSV = os.path.join(BASE_DIR, 'step31_fillExter.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'step40_interpolated_fastest.parquet')
THREADS = 256
FEATURES = ['flow', 'occupancy', 'speed']

@cuda.jit
def local_kernel(feat, nbr, mask, out, K):
    i = cuda.grid(1)
    if i < feat.size:
        if mask[i]:
            s = float32(0.0)
            for j in range(K):
                s += feat[nbr[i, j]]
            out[i] = s / K
        else:
            out[i] = feat[i]

@cuda.jit
def temporal_kernel(ts, mask, offs, out):
    gid = cuda.grid(1)
    if gid < offs.size - 1:
        start = offs[gid]
        end = offs[gid + 1]
        t0 = ts[start]
        v0 = out[start]
        for idx in range(start + 1, end):
            if mask[idx]:
                nxt = idx + 1
                while nxt < end and mask[nxt]:
                    nxt += 1
                if nxt < end:
                    r = (ts[idx] - t0) / (ts[nxt] - t0)
                    out[idx] = v0 + (out[nxt] - v0) * r
                else:
                    out[idx] = v0
            else:
                t0 = ts[idx]
                v0 = out[idx]


def process_partition(pdf, nbr_map, K):
    # merge neighbor indices
    pdf = pdf.merge(nbr_map, on='station_id', how='left')
    # build mask flag
    for m in ['mask_logic', 'mask_md', 'mask_hf']:
        pdf[m] = pdf[m].fillna(False).astype('bool')
    pdf['mask_flag'] = pdf['mask_logic'] | pdf['mask_md'] | pdf['mask_hf']
    N = len(pdf)

    # prepare neighbor list and mask missing per row
    nbrs_list = pdf['nbr_idx'].to_arrow().to_pylist()
    mask_no_nbr = [x is None for x in nbrs_list]
    # split into valid and invalid rows
    idx_good = [i for i, bad in enumerate(mask_no_nbr) if not bad]
    idx_bad  = [i for i, bad in enumerate(mask_no_nbr) if bad]

    # Process only valid subset
    pdf_good = pdf.iloc[idx_good].reset_index(drop=True)
    pdf_bad  = pdf.iloc[idx_bad].reset_index(drop=True)

    # perform local + global + temporal on pdf_good
    # Local interpolation
    coords_nbr = [nbrs_list[i] for i in idx_good]
    nbrs = np.vstack(coords_nbr).astype(np.int32)
    blocks = (len(pdf_good) + THREADS - 1) // THREADS
    d_nbr = cuda.to_device(nbrs)
    mask_gpu = cuda.to_device(pdf_good['mask_flag'].to_numpy())
    for feat in FEATURES:
        arr = pdf_good[feat].fillna(0).to_numpy(dtype=np.float32)
        d_feat = cuda.to_device(arr)
        d_out = cuda.device_array_like(d_feat)
        local_kernel[blocks, THREADS](d_feat, d_nbr, mask_gpu, d_out, K)
        cuda.synchronize()
        pdf_good[feat] = d_out.copy_to_host()

    # Global interpolation
    grp = pdf_good[~pdf_good['mask_flag']].groupby(['station_id','direction'])[FEATURES].mean().reset_index()
    pdf_good = pdf_good.merge(grp, on=['station_id','direction'], how='left', suffixes=('','_g'))
    for feat in FEATURES:
        pdf_good[feat] = pdf_good[f'{feat}_g'].where(pdf_good['mask_flag'], pdf_good[feat])
        pdf_good = pdf_good.drop(columns=[f'{feat}_g'])

    # Temporal interpolation
    pdf_good = pdf_good.sort_values(['station_id','direction','timestamp']).reset_index(drop=True)
    offs = [0]
    for i in range(1, len(pdf_good)):
        if (pdf_good.loc[i,'station_id'], pdf_good.loc[i,'direction'])!= \
           (pdf_good.loc[i-1,'station_id'], pdf_good.loc[i-1,'direction']):
            offs.append(i)
    offs.append(len(pdf_good))
    off_arr = np.array(offs, dtype=np.int32)
    d_ts = cuda.to_device(pdf_good['timestamp'].astype('int64').to_numpy().astype(np.float32))
    d_mask = cuda.to_device(pdf_good['mask_flag'].to_numpy())
    d_offs = cuda.to_device(off_arr)
    tblocks = (off_arr.size - 1 + THREADS - 1) // THREADS
    for feat in FEATURES:
        arr = pdf_good[feat].fillna(0).to_numpy(dtype=np.float32)
        d_arr = cuda.to_device(arr)
        temporal_kernel[tblocks, THREADS](d_ts, d_mask, d_offs, d_arr)
        cuda.synchronize()
        pdf_good[feat] = d_arr.copy_to_host()

    # combine back bad rows (left unmodified)
    pdf_out = cudf.concat([pdf_good, pdf_bad]).sort_index()
    return pdf_out.drop(columns=['mask_flag', 'nbr_idx'])

if __name__ == '__main__':
    t_start = time.time()
    print('DEBUG: starting cluster')
    cluster = LocalCUDACluster(n_workers=5, CUDA_VISIBLE_DEVICES='0,1,2,3,4', dashboard_address=':8787')
    client = Client(cluster)
    print(f'DEBUG: Dashboard at {client.dashboard_link}')
    # prepare neighbor map
    st = pd.read_csv(STATIONS_CSV, usecols=['station_id','latitude','longitude']).dropna().reset_index(drop=True)
    coords = st[['latitude','longitude']].to_numpy(dtype=np.float32)
    K = 8
    nn = NearestNeighbors(n_neighbors=K)
    nn.fit(cp.asarray(coords))
    nbrs = nn.kneighbors(cp.asarray(coords), return_distance=False)
    nbr_map = cudf.DataFrame({'station_id': st['station_id'], 'nbr_idx': cp.asnumpy(nbrs).tolist()})
    # sample metadata
    raw_pdf = pd.read_csv(RAW_CSV, nrows=1, parse_dates=['timestamp'])
    mask_pdf = pd.read_parquet(MASK_PARQUET, columns=['timestamp','station_id','mask_logic','mask_md','mask_hf']).head(1)
    sample_pdf = raw_pdf.merge(mask_pdf, on=['timestamp','station_id'], how='left')
    sample_meta = cudf.DataFrame.from_pandas(sample_pdf)
    sample_meta = process_partition(sample_meta, nbr_map, K)
    # build Dask DataFrame
    raw_ddf = dask_cudf.read_csv(RAW_CSV, parse_dates=['timestamp'])
    mask_ddf = dask_cudf.read_parquet(MASK_PARQUET)[['timestamp','station_id','mask_logic','mask_md','mask_hf']]
    ddf = raw_ddf.merge(mask_ddf, on=['timestamp','station_id'], how='left')
    ddf = ddf.repartition(npartitions=ddf.npartitions)
    out = ddf.map_partitions(
        process_partition,
        nbr_map,
        K,
        meta=sample_meta,
        align_dataframes=False
    )
    with ProgressBar():
        out = out.persist()
    delayed_parts = out.to_delayed()
    futures = client.compute(delayed_parts)
    progress(futures)
    for i, fut in enumerate(futures):
        df = fut.result()
        df.to_parquet(f"{OUTPUT_DIR}/part-{i}.parquet")
    print(f'DEBUG: total time={(time.time()-t_start):.2f}s')
"""""


#!/usr/bin/env python3
# coding: utf-8
"""
step40_fill_fastest_optimized.py

note step40_interpolated_fastest.py note:
  - note CSV note blocksize, note
  - Local/temporal CUDA kernels note 512 threads
  - Global note cuDF-groupby note GPU noterows
  - note to_parquet note Dask note
"""
import os, time
import numpy as np
import cudf, cupy as cp
from numba import cuda, float32
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, progress
import dask_cudf
from cuml.neighbors import NearestNeighbors

BASE_DIR     = '/scratch/lgong1/finalproject/pems_data'
RAW_CSV      = os.path.join(BASE_DIR, 'step31_fillExter.csv')
MASK_PARQ    = os.path.join(BASE_DIR, 'step34_maskMix.parquet')
META_CSV     = os.path.join(BASE_DIR, 'step01_d07_meta.csv')
OUTPUT_PARQ  = os.path.join(BASE_DIR, 'step40_fastest.parquet')

# noterowsnote, CUDA configuration
N_GPUS       = 5
THREADS      = 512              # note block notethread count
FEATURES     = ['flow','occupancy','speed']
K            = 8

@cuda.jit
def local_kernel(feat, nbr, mask, out):
    i = cuda.grid(1)
    if i < feat.size:
        if mask[i]:
            s = float32(0.0)
            for j in range(K):
                s += feat[nbr[i, j]]
            out[i] = s / K
        else:
            out[i] = feat[i]

@cuda.jit
def temporal_kernel(ts, mask, offs, out):
    gid = cuda.grid(1)
    if gid < offs.size - 1:
        start = offs[gid]; end = offs[gid+1]
        t0 = ts[start]; v0 = out[start]
        for idx in range(start+1, end):
            if mask[idx]:
                nxt = idx+1
                while nxt<end and mask[nxt]:
                    nxt += 1
                if nxt<end:
                    r = (ts[idx]-t0)/(ts[nxt]-t0)
                    out[idx] = v0 + (out[nxt]-v0)*r
                else:
                    out[idx] = v0
            else:
                t0 = ts[idx]; v0 = out[idx]

def process_partition(pdf, nbr_map):
    # 1) merge nbr_idx, note mask_flag
    pdf = pdf.merge(nbr_map, on='station_id', how='left')
    for m in ['mask_logic','mask_md','mask_hf']:
        pdf[m] = pdf[m].fillna(False)
    pdf['mask_flag'] = pdf['mask_logic'] | pdf['mask_md'] | pdf['mask_hf']

    # 2) local note
    nbr_list = pdf['nbr_idx'].to_arrow().to_pylist()
    valid = [i for i, x in enumerate(nbr_list) if x is not None]
    bad   = [i for i, x in enumerate(nbr_list) if x is None]
    good  = pdf.iloc[valid].reset_index(drop=True)
    baddf = pdf.iloc[bad]

    nbrs = np.vstack([nbr_list[i] for i in valid]).astype(np.int32)
    blocks = (len(good) + THREADS - 1) // THREADS
    d_nbr  = cuda.to_device(nbrs)
    mask_d = cuda.to_device(good['mask_flag'].to_numpy())
    for feat in FEATURES:
        arr = good[feat].fillna(0).to_numpy(np.float32)
        d_in  = cuda.to_device(arr)
        d_out = cuda.device_array_like(d_in)
        local_kernel[blocks,THREADS](d_in, d_nbr, mask_d, d_out)
        cuda.synchronize()
        good[feat] = d_out.copy_to_host()

    # 3) global note: note cuDF note groupby
    gdf = good.drop(columns=['nbr_idx'])
    # note mask noterows, note (station_id,direction) note
    mean_df = gdf[~gdf['mask_flag']].groupby(['station_id','direction'])[FEATURES].mean().reset_index()
    gdf = gdf.merge(mean_df, on=['station_id','direction'], how='left', suffixes=('','_g'))
    for feat in FEATURES:
        gdf[feat] = gdf['mask_flag'].map_partitions(
            lambda s, orig, fill: orig.where(~s, fill),
            gdf[feat], gdf[f'{feat}_g']
        )
        gdf = gdf.drop(columns=[f'{feat}_g'])

    # 4) temporal note
    gdf = gdf.sort_values(['station_id','direction','timestamp']).reset_index(drop=True)
    # note offsets
    offs=[0]
    for i in range(1,len(gdf)):
        if (gdf.loc[i,'station_id'],gdf.loc[i,'direction'])!= \
           (gdf.loc[i-1,'station_id'],gdf.loc[i-1,'direction']):
            offs.append(i)
    offs.append(len(gdf))
    off_arr = np.array(offs,np.int32)
    blocks_t= (len(offs)-1+THREADS-1)//THREADS
    ts_d   = cuda.to_device(gdf['timestamp'].astype('int64').to_numpy().astype(np.float32))
    mask_d = cuda.to_device(gdf['mask_flag'].to_numpy())
    offs_d  = cuda.to_device(off_arr)
    for feat in FEATURES:
        arr = gdf[feat].fillna(0).to_numpy(np.float32)
        darr= cuda.to_device(arr)
        temporal_kernel[blocks_t,THREADS](ts_d,mask_d,offs_d,darr)
        cuda.synchronize()
        gdf[feat] = darr.copy_to_host()

    # note bad rows
    out = cudf.concat([gdf, baddf]).sort_index()
    return out.drop(columns=['mask_flag','nbr_idx'])

if __name__=='__main__':
    t0 = time.time()

    # note Dask-CUDA
    cluster = LocalCUDACluster(
        n_workers=N_GPUS,
        CUDA_VISIBLE_DEVICES=','.join(map(str,range(N_GPUS))),
        threads_per_worker=1,
        memory_limit='16GB'
    )
    client = Client(cluster)

    # note
    st = cudf.read_csv(META_CSV, usecols=['station_id','latitude','longitude']).dropna().reset_index(drop=True)
    coords = st[['latitude','longitude']].to_pandas().to_numpy(np.float32)
    nn = NearestNeighbors(n_neighbors=K)
    nn.fit(cp.asarray(coords))
    nbrs = nn.kneighbors(cp.asarray(coords), return_distance=False)
    nbr_map = cudf.DataFrame({
        'station_id': st['station_id'],
        'nbr_idx':    cp.asnumpy(nbrs).tolist()
    })

    # readnote + mask data
    usecols = ['timestamp','station_id','direction'] + FEATURES + ['mask_logic','mask_md','mask_hf']
    raw = dask_cudf.read_csv(
        RAW_CSV, usecols=usecols, parse_dates=['timestamp'],
        blocksize='128MB'
    )
    mask = dask_cudf.read_parquet(MASK_PARQ, columns=['timestamp','station_id','mask_logic','mask_md','mask_hf'])
    ddf  = raw.merge(mask, on=['timestamp','station_id'], how='left')
    # noterowsnote: note GPU note 10 note
    ddf = ddf.repartition(npartitions=N_GPUS * 10)

    # map_partitions,  note
    out_ddf = ddf.map_partitions(
        process_partition, nbr_map,
        meta=ddf._meta,
        align_dataframes=False
    )

    # note Parquet, note Dask noterows
    out_ddf.to_parquet(OUTPUT_PARQ, write_index=False)

    client.close()
    print(f"Total elapsed: {time.time()-t0:.1f}s")
