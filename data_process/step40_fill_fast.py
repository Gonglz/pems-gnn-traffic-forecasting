#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step40_fill_fast.py — 三步插值流水线（GPU 加速 + 分块 + Merge MD 掩码）

用法：
  cd finalproject/data_process
  python step40_fill_fast.py [--test-n N]

依赖：
  pandas numpy cudf cupy scikit-learn tqdm numba

说明：
 1. 预先在 CPU 上对所有站点做 KDTree，丢弃经纬度缺失的站点
 2. 将原始长表分块读取，每块：
    • 合并三种掩码（逻辑、MD、健康度）
    • 用 GPU + CuPy 向量化完成 Local / Global / Temporal 三步插值
 3. 输出最终无缺失的 step40_interpolated.csv
 /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step40_fill_fast.py
▶ CPU 预计算站点邻居…
读取到 4883 个站点
NaN 检查：
 latitude     0
longitude    0
dtype: int64
Processing chunks: 325it [6:20:44, 70.29s/it]
✅ 插值完成，结果已保存到 /scratch/lgong1/finalproject/pems_data/step40_interpolated.csv

进程已结束，退出代码为 0

"""

import os
import argparse
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from sklearn.neighbors import KDTree as SKKDTree
from tqdm import tqdm

# ——— 路径 & 参数 ———
BASE       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pems_data'))
RAW_LONG   = os.path.join(BASE, 'step31_fillExter.csv')
LOGIC_CSV  = os.path.join(BASE, 'step30_logic_mask_continuous.csv')
MD_CSV     = os.path.join(BASE, 'step32_md.csv')
HF_CSV     = os.path.join(BASE, 'day_health_factor.csv')
OUT_CSV    = os.path.join(BASE, 'step40_interpolated.csv')

HF_THRESH  = 0.5
LOCAL_K    = 5
CHUNK_SIZE = 500_000

# ——— 预加载小表 ———
# 逻辑掩码
logic_full = pd.read_csv(LOGIC_CSV, parse_dates=['timestamp'])
logic_full = logic_full[['timestamp','station_id','mask_logic']]

# Mahalanobis 掩码
md_full = pd.read_csv(MD_CSV, parse_dates=['timestamp'])
md_full = md_full[['timestamp','station_id','mask_md']]

# 日健康因子
hf_pdf = pd.read_csv(HF_CSV, parse_dates=['date'])
if 'health_factor' not in hf_pdf.columns:
    hf_pdf = hf_pdf.rename(columns={hf_pdf.columns[-1]:'health_factor'})
hf_pdf['date'] = hf_pdf['date'].dt.date
hf_pdf = hf_pdf[['station_id','date','health_factor']]

# ——— CUDA Kernel for Local 插值 ———
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
    # 合并逻辑掩码
    pdf = pdf.merge(logic_full, on=['timestamp','station_id'], how='left')
    pdf['mask_logic'] = pdf['mask_logic'].fillna(False)

    # 合并 Mahalanobis 掩码
    pdf = pdf.merge(md_full, on=['timestamp','station_id'], how='left')
    pdf['mask_md'] = pdf['mask_md'].fillna(False)

    # 合并健康度掩码
    pdf['date'] = pdf['timestamp'].dt.date
    pdf = pdf.merge(hf_pdf, on=['station_id','date'], how='left')
    pdf['health_factor'] = pdf['health_factor'].fillna(1.0)
    pdf['mask_hf'] = pdf['health_factor'] < HF_THRESH
    pdf.drop(columns=['date'], inplace=True)

    # 总掩码
    pdf['mask'] = pdf['mask_logic'] | pdf['mask_md'] | pdf['mask_hf']
    mask_np = pdf['mask'].to_numpy()

    # —— GPU Local 插值 ——
    # 1) 准备输入数组
    N = len(pdf)
    coords = pdf[['latitude','longitude']].to_numpy()
    # 2) 构建每行的邻居索引表
    # 站点ID -> 在 neighbor_map 中的位置
    sid_to_pos = {sid: pos for pos, sid in enumerate(neighbor_map['station_ids'])}
    nbr_idx = np.zeros((N, LOCAL_K), dtype=np.int32)

    for i, sid in enumerate(pdf['station_id']):
        pos = sid_to_pos.get(sid, None)
        if pos is not None:
            # 只有真正算得出邻居的位置，才填入 nbr_idx
            nbr_idx[i, :] = neighbor_map['neighbors'][pos]
        # else: 不在预计算表里的站点，nbr_idx[i] 保持 [0,0,0,0,0]

    # 3) 对每个特征跑 kernel
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

    # —— GPU Global 插值 via cuDF ——
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

    # —— GPU Temporal 插值 via CuPy interp ——
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

    # 三步优先级合并：Local -> Global -> Temporal
    for feat in ['flow','occupancy','speed']:
        pdf[feat] = np.where(pdf['mask'], local_out[feat], pdf[feat])
        pdf[feat] = np.where(pdf['mask'], global_out[feat], pdf[feat])
        pdf[feat] = np.where(pdf['mask'], temporal_out[feat], pdf[feat])

    # 写入
    pdf.to_csv(OUT_CSV,
               index=False,
               mode='w' if is_first else 'a',
               header=is_first)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=0, help='测试模式：前 N 行')
    args = parser.parse_args()

    # ——— CPU 预计算站点邻居 ———
    print("▶ CPU 预计算站点邻居…")
    meta = pd.read_csv(RAW_LONG, usecols=['station_id','latitude','longitude'])
    station_meta = (meta
        .drop_duplicates('station_id')
        .dropna(subset=['latitude','longitude'])
        .reset_index(drop=True))
    print(f"读取到 {len(station_meta)} 个站点")
    print("NaN 检查：\n", station_meta[['latitude','longitude']].isnull().sum())

    coords = station_meta[['latitude','longitude']].to_numpy(dtype=np.float32)
    assert not np.isnan(coords).any(), "coords 中仍有 NaN！"
    sk = SKKDTree(coords)
    nbr = sk.query(coords, k=LOCAL_K+1, return_distance=False)[:,1:]

    # 把 station_id 和 对应的 neighbor position 存到一个 dict
    neighbor_map = {
      'station_ids': station_meta['station_id'].to_numpy(),
      'neighbors' : nbr
    }

    # ——— 分块或 test 模式 处理 ———
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

    print("✅ 插值完成，结果已保存到", OUT_CSV)

if __name__ == '__main__':
    main()
