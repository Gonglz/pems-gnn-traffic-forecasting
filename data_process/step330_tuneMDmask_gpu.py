#!/usr/bin/env python3
# coding: utf-8
"""
step31_md_cuda.py

功能：
 1. 加载扩展外部特征数据（flow, occupancy, speed, tavg, pcpn）及分组标签 (is_weekend, is_holiday, in_custom_event)
 2. 按分组标签分组，计算每组的均值 μₖ 和协方差 Σₖ（仅针对连续特征），并加微量正则化后求逆 Σₖ⁻¹
 3. 在多 GPU 环境下并行计算每条记录的 Mahalanobis 距离平方
 4. 对每组使用 95% 分位自适应阈值选取，生成 mask_md
 5. 输出带 md_squared 和 mask_md 的 step32_md.csv

依赖：
  pip install pandas numpy numba pyarrow

用法：
  cd finalproject/data_process
  python step31_md_cuda.py
"""
import math
import numpy as np
import pandas as pd
from numba import cuda, float32

# 连续特征维度（不包含布尔变量）
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
    print('加载数据并构建分组标签...')
    df = pd.read_csv('../pems_data/step31_fillExter.csv', usecols=CONT_FEATURES + ['is_weekend', 'is_holiday', 'in_custom_event'])
    # 丢弃包含 NaN 的记录，确保后续计算无缺失
    df = df.dropna(subset=CONT_FEATURES).reset_index(drop=True)
    # 分组标签
    grp = (df['is_holiday'].astype(int) * 4 +
           df['in_custom_event'].astype(int) * 2 +
           df['is_weekend'].astype(int))
    df['group'] = grp
    N = len(df)
    print(f'共 {N} 条记录，连续特征维度 {D}')

    # 结果容器
    md_sq = np.zeros(N, dtype=np.float32)
    mask_md = np.zeros(N, dtype=bool)
    thresholds = {}  # 存储每个分组的阈值(N, dtype=bool)

    devices = list(cuda.gpus)
    print(f'检测到 {len(devices)} 张 GPU')

    # 对每个组并行处理
    for label in sorted(df['group'].unique()):
        idx = np.where(df['group'] == label)[0]
        size = len(idx)
        print(f'组 {label}: {size} 条')
        if size < D + 1:
            print('  样本不足，跳过')
            continue
        Xg = df.iloc[idx][CONT_FEATURES].values.astype(np.float32)
        # 对缺失值按组均值进行填充，避免 NaN 导致协方差和距离全为 NaN
        mu = np.nanmean(Xg, axis=0).astype(np.float32)
        # 将 NaN 替换为对应维度的组均值
        inds = np.where(np.isnan(Xg))
        if inds[0].size > 0:
            Xg[inds] = np.take(mu, inds[1])
        # 计算协方差并加正则化，保证可逆
        cov = np.cov(Xg, rowvar=False).astype(np.float32)
        cov += np.eye(D, dtype=np.float32) * 1e-6
        inv_cov = np.linalg.inv(cov).astype(np.float32)
        dist_g = np.zeros(size, dtype=np.float32)

        # GPU 并行
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
        thresholds[label] = thr  # 记录阈值
        print(f'  阈值(95%): {thr:.4f}')
        md_sq[idx] = dist_g
        mask_md[idx] = dist_g > thr

    print('写入结果...')
    out = df.copy()
    out['md_squared'] = md_sq
    out['mask_md'] = mask_md
    out.to_csv('../pems_data/step32_md.csv', index=False)
    print('完成: ../pems_data/step32_md.csv')

    # 可视化每个分组的阈值
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
        print(f'✔ 阈值可视化已保存: {fig_path}')
    except Exception as e:
        print(f'⚠ 阈值可视化失败: {e}')

if __name__ == '__main__':
    main()
