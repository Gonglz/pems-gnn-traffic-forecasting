#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step40_fill_fastest.py

Optimized 5‑minute interpolation using pure CUDA kernels for local, global and
temporal fills, orchestrated via Dask‑cuDF over multiple GPUs.

/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data\_process/fillTest.py
2025-05-05 16:03:28,610 - distributed.preloading - INFO - Creating preload: dask\_cuda.initialize
2025-05-05 16:03:28,610 - distributed.preloading - INFO - Import preload module: dask\_cuda.initialize
2025-05-05 16:03:28,704 - distributed.preloading - INFO - Creating preload: dask\_cuda.initialize
2025-05-05 16:03:28,704 - distributed.preloading - INFO - Import preload module: dask\_cuda.initialize
2025-05-05 16:03:28,706 - distributed.preloading - INFO - Creating preload: dask\_cuda.initialize
2025-05-05 16:03:28,706 - distributed.preloading - INFO - Import preload module: dask\_cuda.initialize
2025-05-05 16:03:28,881 - distributed.preloading - INFO - Creating preload: dask\_cuda.initialize
2025-05-05 16:03:28,881 - distributed.preloading - INFO - Import preload module: dask\_cuda.initialize
2025-05-05 16:08:20,678 - distributed.utils\_perf - WARNING - full garbage collections took 18% CPU time recently (threshold: 10%)
2025-05-05 16:08:49,688 - distributed.utils\_perf - WARNING - full garbage collections took 20% CPU time recently (threshold: 10%)
2025-05-05 16:09:23,507 - distributed.utils\_perf - WARNING - full garbage collections took 15% CPU time recently (threshold: 10%)
2025-05-05 16:09:26,874 - distributed.utils\_perf - WARNING - full garbage collections took 15% CPU time recently (threshold: 10%)
2025-05-05 16:09:30,976 - distributed.utils\_perf - WARNING - full garbage collections took 15% CPU time recently (threshold: 10%)
2025-05-05 16:09:36,081 - distributed.utils\_perf - WARNING - full garbage collections took 15% CPU time recently (threshold: 10%)
2025-05-05 16:09:42,827 - distributed.utils\_perf - WARNING - full garbage collections took 16% CPU time recently (threshold: 10%)
2025-05-05 16:09:51,321 - distributed.utils\_perf - WARNING - full garbage collections took 16% CPU time recently (threshold: 10%)
2025-05-05 16:10:01,878 - distributed.utils\_perf - WARNING - full garbage collections took 16% CPU time recently (threshold: 10%)
2025-05-05 16:10:14,773 - distributed.utils\_perf - WARNING - full garbage collections took 16% CPU time recently (threshold: 10%)
2025-05-05 16:10:30,718 - distributed.utils\_perf - WARNING - full garbage collections took 19% CPU time recently (threshold: 10%)
2025-05-05 16:10:31,203 - distributed.utils\_perf - WARNING - full garbage collections took 17% CPU time recently (threshold: 10%)
2025-05-05 16:10:50,386 - distributed.utils\_perf - WARNING - full garbage collections took 20% CPU time recently (threshold: 10%)
2025-05-05 16:10:51,552 - distributed.utils\_perf - WARNING - full garbage collections took 17% CPU time recently (threshold: 10%)
2025-05-05 16:11:18,950 - distributed.utils\_perf - WARNING - full garbage collections took 20% CPU time recently (threshold: 10%)
2025-05-05 16:11:20,817 - distributed.utils\_perf - WARNING - full garbage collections took 17% CPU time recently (threshold: 10%)
2025-05-05 16:11:49,053 - distributed.utils\_perf - WARNING - full garbage collections took 20% CPU time recently (threshold: 10%)
2025-05-05 16:11:52,366 - distributed.utils\_perf - WARNING - full garbage collections took 17% CPU time recently (threshold: 10%)
2025-05-05 16:12:27,149 - distributed.utils\_perf - WARNING - full garbage collections took 20% CPU time recently (threshold: 10%)
2025-05-05 16:12:32,941 - distributed.utils\_perf - WARNING - full garbage collections took 18% CPU time recently (threshold: 10%)
2025-05-05 16:13:15,260 - distributed.utils\_perf - WARNING - full garbage collections took 21% CPU time recently (threshold: 10%)
Total time: 658.4399013519287 s

进程已结束，退出代码为 0

"""
import os, sys, time
import numpy as np
import cudf, cupy as cp, dask_cudf
from numba import cuda, float32
from cuml.neighbors import NearestNeighbors
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR     = '/scratch/lgong1/finalproject/pems_data'
RAW_PARQ     = os.path.join(BASE_DIR, 'step31_fillExter.parquet')
MASK_PARQ    = os.path.join(BASE_DIR, 'step34_maskMix.parquet')
STATIONS_CSV = os.path.join(BASE_DIR, 'step01_d07_meta.csv')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'step40_interpolated_fastest.parquet')
THREADS      = 256
FEATURES     = ['flow','occupancy','speed']
K_NEIGHBORS  = 8
GPU_DEVICES  = '0,1,2,3'
NUM_WORKERS  = 4
REPARTITIONS = 2

# ─── CUDA kernels ────────────────────────────────────────────────────────────

@cuda.jit
def local_kernel(feat, nbrs, mask_flag, out):
    i = cuda.grid(1)
    if i < feat.shape[0]:
        if mask_flag[i]:
            acc = float32(0.0); cnt = 0
            for j in range(K_NEIGHBORS):
                nb = nbrs[i,j]
                if 0 <= nb < feat.shape[0] and feat[nb] > 0:
                    acc += feat[nb]; cnt += 1
            out[i] = acc/cnt if cnt>0 else feat[i]
        else:
            out[i] = feat[i]

@cuda.jit
def global_kernel(feat, mask_flag, sid, did, grp_idx, grp_val, G, out):
    i = cuda.grid(1)
    if i < feat.shape[0] and mask_flag[i]:
        s = sid[i]; d = did[i]
        for g in range(G):
            if grp_idx[g,0]==s and grp_idx[g,1]==d:
                out[i] = grp_val[g]
                break

@cuda.jit
def temporal_kernel(ts, mask_flag, offs, out):
    seg = cuda.blockIdx.x
    start = offs[seg]; end = offs[seg+1]
    tid = cuda.threadIdx.x; stride = cuda.blockDim.x
    for idx in range(start+tid, end, stride):
        if mask_flag[idx]:
            prev = idx-1
            while prev>=start and mask_flag[prev]: prev-=1
            nxt = idx+1
            while nxt<end and mask_flag[nxt]: nxt+=1
            if prev>=start and nxt<end:
                t0,t1 = ts[prev],ts[nxt]; v0,v1=out[prev],out[nxt]
                r=(ts[idx]-t0)/(t1-t0)
                out[idx]=v0+(v1-v0)*r
            elif prev>=start:
                out[idx]=out[prev]
            elif nxt<end:
                out[idx]=out[nxt]

# ─── Neighbor map creation ───────────────────────────────────────────────────

def build_nbr_map():
    df = cudf.read_csv(STATIONS_CSV,usecols=['station_id','latitude','longitude'])\
             .dropna().reset_index(drop=True)
    coords = df[['latitude','longitude']].to_pandas().values.astype('float32')
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(cp.asarray(coords))
    nbrs = nn.kneighbors(cp.asarray(coords), return_distance=False)
    return cudf.DataFrame({
        'station_id': df['station_id'],
        'nbr_idx':    cp.asnumpy(nbrs).tolist()
    })

# ─── Per-partition interpolation ─────────────────────────────────────────────

def process_partition(pdf, nbr_map):
    if pdf.shape[0]==0:
        return pdf
    # 1) 合并邻居映射
    pdf = pdf.merge(nbr_map,on='station_id',how='left')
    # 2) Mask 列 OR
    pdf['mask_flag'] = (pdf['mask_logic']|pdf['mask_md']|pdf['mask_hf']).fillna(False)
    # 3) 类别编码 direction -> dir_code
    pdf['dir_code'] = pdf['direction'].astype('category').cat.codes.astype('int32')

    # 拆分有无 nbr
    nbrs_list = pdf['nbr_idx'].to_arrow().to_pylist()
    no_nbr    = np.array([x is None for x in nbrs_list],dtype=bool)
    idx_good  = np.where(~no_nbr)[0]; idx_bad = np.where(no_nbr)[0]
    if idx_good.size==0:
        return pdf.drop(columns=['nbr_idx','mask_flag','dir_code'])

    pg = pdf.iloc[idx_good].reset_index(drop=True)
    N  = pg.shape[0]
    blocks = (N+THREADS-1)//THREADS

    # 数据上 GPU
    sid    = pg['station_id'].to_numpy().astype('int32')
    did    = pg['dir_code'].to_numpy().astype('int32')
    mflag  = pg['mask_flag'].to_numpy().astype(bool)
    d_sid  = cuda.to_device(sid)
    d_did  = cuda.to_device(did)
    d_mask = cuda.to_device(mflag)
    nbrs_a = np.vstack(pg['nbr_idx'].to_arrow().to_pylist()).astype('int32')
    d_nbr  = cuda.to_device(nbrs_a)

    # 1) 本地空间插值
    for feat in FEATURES:
        arr = pg[feat].fillna(0).to_numpy('float32')
        d_in  = cuda.to_device(arr)
        d_out = cuda.device_array_like(d_in)
        local_kernel[blocks,THREADS](d_in,d_nbr,d_mask,d_out)
        cuda.synchronize()
        pg[feat]=d_out.copy_to_host()

    # 2) 全局平均插值
    grp = pg[~pg['mask_flag']].groupby(['station_id','dir_code'])[FEATURES]\
            .mean().reset_index()
    if grp.shape[0]>0:
        G = grp.shape[0]
        gi = grp[['station_id','dir_code']].to_numpy('int32')
        gv = grp[FEATURES].to_numpy('float32')
        d_gi = cuda.to_device(gi)
        d_gv = cuda.to_device(gv)
        for fi,feat in enumerate(FEATURES):
            d_f = cuda.to_device(pg[feat].to_numpy('float32'))
            global_kernel[blocks,THREADS](
                d_f,d_mask,d_sid,d_did,d_gi,d_gv[:,fi],G,d_f
            )
            cuda.synchronize()
            pg[feat]=d_f.copy_to_host()

    # 3) 时间线性插值
    pg = pg.sort_values(['station_id','dir_code','timestamp'])\
           .reset_index(drop=True)
    ts = pg['timestamp'].astype('int64').to_numpy('float32')
    st = pg['station_id'].to_numpy()
    dc = pg['dir_code'].to_numpy()
    ch = ((st[1:]!=st[:-1])|(dc[1:]!=dc[:-1]))
    offs = np.concatenate([[0],np.where(ch)[0]+1,[N]]).astype('int32')
    M = offs.size-1
    d_ts   = cuda.to_device(ts)
    d_offs = cuda.to_device(offs)
    for feat in FEATURES:
        d_f = cuda.to_device(pg[feat].to_numpy('float32'))
        temporal_kernel[M,THREADS](d_ts,d_mask,d_offs,d_f)
        cuda.synchronize()
        pg[feat]=d_f.copy_to_host()

    # 合并返还
    pb = pdf.iloc[idx_bad].reset_index(drop=True)
    out = cudf.concat([pg,pb]).sort_index()
    return out.drop(columns=['nbr_idx','mask_flag','dir_code'])

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    for p in (RAW_PARQ,MASK_PARQ,STATIONS_CSV):
        if not os.path.exists(p):
            print(f"Missing {p}",file=sys.stderr); sys.exit(1)

    with LocalCUDACluster(
        n_workers=NUM_WORKERS,
        CUDA_VISIBLE_DEVICES=GPU_DEVICES,
        threads_per_worker=1
    ) as cluster, Client(cluster) as client:

        nbr_map = build_nbr_map()
        nbr_map = client.scatter(nbr_map, broadcast=True)

        raw = dask_cudf.read_parquet(
            RAW_PARQ,
            columns=['timestamp','station_id','direction']+FEATURES
        )
        mask= dask_cudf.read_parquet(
            MASK_PARQ,
            columns=['timestamp','station_id','mask_logic','mask_md','mask_hf']
        )
        ddf = raw.merge(mask,on=['timestamp','station_id'],how='left')
        npart = max(ddf.npartitions*REPARTITIONS,NUM_WORKERS*2)
        ddf = ddf.repartition(npartitions=npart)

        meta = ddf._meta
        meta = process_partition(meta,nbr_map)

        out = ddf.map_partitions(
            process_partition,nbr_map,
            meta=meta,align_dataframes=False
        ).persist()

        out.to_parquet(OUTPUT_DIR,write_index=False)

if __name__=='__main__':
    t0=time.time(); main()
    print("Total time:",time.time()-t0,"s")
