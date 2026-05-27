#!/usr/bin/env python3
# coding: utf-8
"""
step32_mdMask.py

功能：
 1. 加载扩展外部特征数据（flow, occupancy, speed, tavg, pcpn）及分组标签 (is_weekend, is_holiday, in_custom_event)
 2. 按分组标签分组，计算每组的均值 μₖ 和协方差 Σₖ（仅针对连续特征），并加微量正则化后求逆 Σₖ⁻¹
 3. 在多 GPU 环境下并行计算每条记录的 Mahalanobis 距离平方
 4. 对每组使用自适应“膝点”检测法自动选取阈值，生成 mask_md
 5. 输出带 md_squared、mask_md 的 step32_md.csv，并可视化阈值分布图

依赖：
  pip install pandas numpy numba pyarrow matplotlib

用法：
  cd finalproject/data_process
  python step32_MDmask.py

注意：
 - 本脚本中用 @cuda.jit 注解的方法 mahalanobis_kernel 即为 CUDA kernel，并行计算距离。
 - 结果包括：
     - step32_MDmask.csv：原始索引、md_squared、mask_md
     - step32_thresholds.png：各组阈值柱状图
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step32_MDmask.py
step32_MDmask: 加载数据并构建分组标签...
共 162169176 条记录，连续特征维度 5
检测到 5 张 GPU
组 0: 116842752 条
  自适应阈值(膝点): 35.4626
组 1: 45326424 条
  自适应阈值(膝点): 38.8974
写入 step32_md.csv...
完成: ../pems_data/step32_md.csv
✔ 阈值可视化已保存: ../pems_data/step32_thresholds.png

进程已结束，退出代码为 0

进程已结束，退出代码为 0
"""
import math
import numpy as np
import pandas as pd
from numba import cuda, float32

# 连续特征维度（不包含布尔变量）
CONT_FEATURES = ['flow', 'occupancy', 'speed', 'tavg', 'pcpn']
D = len(CONT_FEATURES)

# ---------------------------
# CUDA Kernel
# 使用 Numba @cuda.jit 装饰的函数即为 CUDA kernel
# ---------------------------
@cuda.jit
def mahalanobis_kernel(X, mu, inv_cov, distances):
    i = cuda.grid(1)
    n = X.shape[0]
    if i < n:
        # 局部数组 tmp 存储中心化向量
        tmp = cuda.local.array(D, float32)
        for j in range(D):
            tmp[j] = X[i, j] - mu[j]
        # 计算 Σ⁻¹ * tmp
        y = cuda.local.array(D, float32)
        for j in range(D):
            acc = 0.0
            for k in range(D):
                acc += inv_cov[j, k] * tmp[k]
            y[j] = acc
        # 计算 Mahalanobis 距离平方
        dist2 = 0.0
        for j in range(D):
            dist2 += tmp[j] * y[j]
        distances[i] = dist2


def main():
    print('step32_MDmask: 加载数据并构建分组标签...')
    df = pd.read_csv(
        '../pems_data/step31_fillExter.csv',
        usecols=CONT_FEATURES
                + ['is_weekend', 'is_holiday', 'in_custom_event']
                + ['station_id', 'timestamp']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 丢弃 NaN 记录
    # 对连续特征先做简单填补，再保留所有行
    means = df[CONT_FEATURES].mean()
    df[CONT_FEATURES] = df[CONT_FEATURES].fillna(means)

    # 构建分组标签：holiday*4 + event*2 + weekend*1
    grp = (df['is_holiday'].astype(int) * 4 +
           df['in_custom_event'].astype(int) * 2 +
           df['is_weekend'].astype(int))
    df['group'] = grp
    N = len(df)
    print(f'共 {N} 条记录，连续特征维度 {D}')

    # 准备结果容器
    md_sq = np.zeros(N, dtype=np.float32)
    mask_md = np.zeros(N, dtype=bool)
    thresholds = {}

    # 检测所有 GPU
    devices = list(cuda.gpus)
    print(f'检测到 {len(devices)} 张 GPU')

    # 分组并行计算
    for label in sorted(df['group'].unique()):
        idx = np.where(df['group'] == label)[0]
        size = len(idx)
        print(f'组 {label}: {size} 条')
        if size < D + 1:
            print('  样本不足，跳过')
            continue
        # 提取组内数据
        Xg = df.iloc[idx][CONT_FEATURES].values.astype(np.float32)
        # 计算组均值并填补剩余 NaN
        mu = np.nanmean(Xg, axis=0).astype(np.float32)
        inds = np.where(np.isnan(Xg))
        if inds[0].size > 0:
            Xg[inds] = np.take(mu, inds[1])
        # 协方差矩阵及正则化
        cov = np.cov(Xg, rowvar=False).astype(np.float32)
        cov += np.eye(D, dtype=np.float32) * 1e-6
        inv_cov = np.linalg.inv(cov).astype(np.float32)
        dist_g = np.zeros(size, dtype=np.float32)

        # 多 GPU 并行
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

        # 自适应阈值（膝点检测）
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
        print(f'  自适应阈值(膝点): {thr:.4f}')

        # 标记结果
        md_sq[idx] = dist_g
        mask_md[idx] = dist_g > thr

    # 输出结果
    print('写入 step32_md.csv...')
    out = pd.DataFrame({
        'station_id': df['station_id'],
        'timestamp': df['timestamp'],
        'md_squared': md_sq,
        'mask_md': mask_md
    })
    out.to_csv('../pems_data/step32_md.csv', index=False)

    print('完成: ../pems_data/step32_md.csv')

    # 可视化阈值
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
        print(f'✔ 阈值可视化已保存: {fig_path}')
    except Exception as e:
        print(f'⚠ 阈值可视化失败: {e}')

if __name__ == '__main__':
    main()