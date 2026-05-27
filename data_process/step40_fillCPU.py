#!/usr/bin/env python3
# coding: utf-8
"""
step40_fill_gpu_demo.py

功能：
 1. 加载清洗输出(step32_fillExter.csv)、逻辑掩码(step3_logic_mask_continuous.csv)、
    MD掩码(step33_md.csv)、日健康因子(day_health_factor.csv)
 2. 合并逻辑掩码(mask_flag)、MD掩码(mask_md)、健康因子掩码(mask_hf)
 3. 对所有掩码点执行四步插值：Local, Global, Temporal, Cluster(占位)
 4. 输出插值后无缺失训练集(step40_interpolated.csv)

依赖：
  pip install pandas numpy scipy scikit-learn tqdm

用法：
  cd finalproject/data_process
  python step40_fill_gpu_demo.py
"""
import os
import pandas as pd
import numpy as np
# 使用 sklearn 的 KDTree 替代 SciPy cKDTree，避免 IDE 报红
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.neighbors import KDTree  # 代替 cKDTree

# 路径配置
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_LONG_CSV   = os.path.join(BASE_DIR, 'pems_data', 'step31_fillExter.csv')  # 含原始流量与外部特征
LOGIC_CSV      = os.path.join(BASE_DIR, 'pems_data', 'step30_logic_mask_continuous.csv')  # 逻辑掩码结果
MD_CSV         = os.path.join(BASE_DIR, 'pems_data', 'step32_md.csv')  # Mahalanobis 掩码结果
HF_CSV         = os.path.join(BASE_DIR, 'pems_data', 'day_health_factor.csv')  # 日健康因子表
OUT_CSV        = os.path.join(BASE_DIR, 'pems_data', 'step40_interpolated.csv')   # 最终插值输出


# 参数
HF_THRESH    = 0.5
LOCAL_K      = 5
TEMP_METHOD  = 'linear'

def load_and_merge():
    # 读取清洗主表
    df = pd.read_csv(RAW_LONG_CSV, parse_dates=['timestamp'])
    # 读取逻辑掩码并重命名
    logic = pd.read_csv(LOGIC_CSV, parse_dates=['timestamp'])
    logic = logic[['timestamp','station_id','mask_logic']]
    logic.rename(columns={'mask_logic':'mask_flag'}, inplace=True)
    df = df.merge(logic, on=['timestamp','station_id'], how='left')
    df['mask_flag'] = df['mask_flag'].fillna(0).astype(int)
    # 读取 MD 掩码
    md = pd.read_csv(MD_CSV)
    df['mask_md'] = md['mask_md']
    # 读取并合并日健康因子
    hf = pd.read_csv(HF_CSV, parse_dates=['date'])
    if 'health_factor' not in hf.columns:
        hf.rename(columns={hf.columns[-1]:'health_factor'}, inplace=True)
    hf['date'] = hf['date'].dt.date
    df['date'] = df['timestamp'].dt.date
    df = df.merge(hf[['station_id','date','health_factor']], on=['station_id','date'], how='left')
    df.drop(columns=['date'], inplace=True)
    df['mask_hf'] = df['health_factor'] < HF_THRESH
    # 生成总掩码
    df['mask'] = (df['mask_flag'] > 0) | df['mask_md'] | df['mask_hf']
    return df


def local_interpolate(df, feats):
    """Local 插值：KDTree + 简单平均"""
    coords = df[['latitude','longitude']].values
    mask_coord = np.isfinite(coords).all(axis=1)
    bad_idx = np.where(df['mask'] & mask_coord)[0]
    good_indices = np.where(mask_coord)[0]
    tree = KDTree(coords[good_indices])
    nbrs = tree.query(coords[bad_idx], k=LOCAL_K+1, return_distance=False)[:,1:]

    out = {}
    for feat in feats:
        vals = df[feat].values
        new = vals.copy()
        for i, idx in enumerate(bad_idx):
            neigh = nbrs[i]
            global_idx = good_indices[neigh]
            new[idx] = np.nanmean(vals[global_idx])
        out[feat] = new
    return out


def global_interpolate(df, feats):
    """Global 插值：按站点+方向全局均值"""
    grp = df.loc[~df['mask']].groupby(['station_id','direction'])
    out = {}
    for feat in feats:
        means = grp[feat].mean()
        new = df[feat].copy()
        bad_idx = np.where(df['mask'])[0]
        for idx in bad_idx:
            key = (df.at[idx,'station_id'], df.at[idx,'direction'])
            new.at[idx] = means.get(key, np.nan)
        out[feat] = new
    return out


def temporal_interpolate(df, feats):
    """Temporal 插值：线性插值"""
    out = {}
    grouped = df.groupby(['station_id','direction'])
    for feat in feats:
        new = df[feat].copy()
        for _, g in tqdm(grouped, desc=f"Temporal {feat}"):
            ts = g['timestamp'].astype(np.int64).values
            vals = g[feat].values
            mask = g['mask'].values
            if mask.all(): continue
            f = interp1d(ts[~mask], vals[~mask], kind=TEMP_METHOD, fill_value='extrapolate')
            new.loc[g.index] = f(ts)
        out[feat] = new
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=0, help='测试模式，前 N 行')
    parser.add_argument('--sample-frac', type=float, default=0.0, help='测试模式，随机比例')
    args = parser.parse_args()

    df = load_and_merge()
    # 测试模式
    if args.test_n > 0:
        print(f"Test mode: first {args.test_n} rows")
        df = df.head(args.test_n).reset_index(drop=True)
    elif 0 < args.sample_frac < 1:
        n = int(len(df)*args.sample_frac)
        print(f"Test mode: random {n} rows ({args.sample_frac*100:.1f}%)")
        df = df.sample(n, random_state=42).reset_index(drop=True)

    print('Loaded', len(df), 'records, masks applied')

    feats = ['flow','occupancy','speed']
    print('Local interpolation...')
    local_map = local_interpolate(df, feats)
    print('Global interpolation...')
    global_map = global_interpolate(df, feats)
    print('Temporal interpolation...')
    temp_map = temporal_interpolate(df, feats)

    # 合并优先级 Local -> Global -> Temporal
    for feat in feats:
        df[feat] = np.where(df['mask'], local_map[feat], df[feat])
        df[feat] = np.where(df['mask'], global_map[feat], df[feat])
        df[feat] = np.where(df['mask'], temp_map[feat], df[feat])

    df.to_csv(OUT_CSV, index=False)
    print('Step40 fill done, saved to', OUT_CSV)

if __name__ == '__main__':
    main()