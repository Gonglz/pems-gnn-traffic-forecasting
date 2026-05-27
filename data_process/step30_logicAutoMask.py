#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step30_logicContinuousTuning.py (Two-Stage Coarse-Fine Search)

功能：
  - 对 Rule5/7/8 连续分数进行两阶段自动调优（粗调 + 细调）
  - 分块抽样与列裁剪：节省内存，限制样本行数
  - 并行计算：Joblib 多核加速
  - 同时评估日级 & 时级 F1-score
  - 输出调优结果、热力图，并在全量数据上应用最优参数生成最终掩膜

Usage:
  python step30_logicContinuousTuning.py

/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step30_logicAutoMask.py
1. 分块读取并抽样...
Sampling: 325it [02:19,  2.34it/s]
  读取 162169176 条, 抽样 16216918 条
  限制样本至 100000 条
2. 标注 & 计算分数...
3. 阶段1: 粗调搜索...
  粗调最优: {'alpha5': 1.0, 'alpha7': 0.0, 'alpha8': 0.0, 'threshold': 0.2, 'F1_day': 0.2978711282450576, 'F1_time': 0.2968994948615224, 'F1_sum': 0.59477062310658}
4. 阶段2: 细调搜索...
  调优结果 & 热力图已保存
5. 最终最优参数: {'alpha5': 0.5, 'alpha7': 0.5, 'alpha8': 0.5, 'threshold': 0.0, 'F1_day': 0.2994507318831643, 'F1_time': 0.2985539527032906, 'F1_sum': 0.5980046845864548}
6. 分块应用最优掩膜...
Apply Mask: 325it [1:29:22, 16.50s/it]
  最终掩膜结果保存: /scratch/lgong1/finalproject/pems_data/step30_logic_mask_continuous.csv

进程已结束，退出代码为 0


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from pathlib import Path
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

# ─── 配置 ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_FILE      = BASE_DIR / 'pems_data' / 'step21_fillHealth.csv'
DET_DIR        = BASE_DIR / 'pems_data' / 'pems_detector'
OUTPUT_TUNE    = BASE_DIR / 'pems_data' / 'step22_continuous_tuning.csv'
OUTPUT_PLOT    = BASE_DIR / 'pems_data' / 'step22_continuous_heatmap.png'
OUTPUT_BEST    = BASE_DIR / 'pems_data' / 'step30_logic_mask_continuous.csv'

# 调参设置
SAMPLE_FRAC    = 0.1       # 抽样比例
MAX_SAMPLES    = 100_000   # 最大样本行数
N_JOBS         = min(multiprocessing.cpu_count(), 8)

# 默认参数范围
def default_alphas():
    return [0.0, 0.5, 1.0]

def default_thresholds():
    return [0.2, 0.5, 0.8]

# ─── 真标签构造 ─────────────────────────────────────────────────────
def load_true_bad(df):
    slice_good = set()
    for f in DET_DIR.glob('Detector Health *.xlsx'):
        d = pd.to_datetime(f.stem.split()[-1], format='%m%d%Y').date()
        dfh = pd.read_excel(f, sheet_name='Report Data')
        slice_good.update((str(s).strip(), d) for s in dfh['VDS'])
    df['date'] = df['timestamp'].dt.date
    df['true_bad'] = df.apply(
        lambda r: 0 if (str(r.station_id), r.date) in slice_good else 1,
        axis=1
    )
    return df

# ─── 连续分数计算 ────────────────────────────────────────────────────
def compute_scores(df):
    gq = df['speed'].quantile(0.99)
    df['station_q'] = df.groupby('station_id')['speed']\
                       .transform(lambda x: x.quantile(0.99))
    df['s5'] = df['speed'] / (0.5 * df['station_q'] + 0.5 * gq)
    sq = df['samples'].quantile(0.10)
    df['s7'] = ((sq - df['samples']) / sq).clip(lower=0)
    df = df.sort_values(['station_id','timestamp'])
    for i in [1, 2]:
        df[f'prev{i}'] = df.groupby('station_id')['flow'].shift(i)
        df[f'next{i}'] = df.groupby('station_id')['flow'].shift(-i)
    df['s8'] = ((df['flow'] == 0) & (df['occupancy'] == 0) &
               (df['prev1'] > 0) & (df['next1'] > 0) &
               (df['prev2'] > 0) & (df['next2'] > 0)).astype(int)
    return df

# ─── 日/时级 F1 评估 ─────────────────────────────────────────────────
def evaluate(df, thresh):
    y_true = df['true_bad'].values
    y_pred = (df['score'] > thresh).astype(int).values
    f1_time = f1_score(y_true, y_pred)
    tmp = df[['station_id','date']].copy()
    tmp['pred'] = y_pred; tmp['true'] = y_true
    day = tmp.groupby(['station_id','date']).agg({'pred':'max','true':'max'})
    f1_day = f1_score(day['true'], day['pred'])
    return f1_day, f1_time

# ─── 单参评估 ─────────────────────────────────────────────────────────
def eval_one(params, df_sample):
    a5, a7, a8, th = params
    df = df_sample.copy()
    df['score'] = a5 * df['s5'] + a7 * df['s7'] + a8 * df['s8']
    return {'alpha5':a5,'alpha7':a7,'alpha8':a8,'threshold':th,
            **dict(zip(['F1_day','F1_time'], evaluate(df, th))),
            'F1_sum': sum(evaluate(df, th))}

# ─── 主流程 ─────────────────────────────────────────────────────────────
def main():
    # 1. 分块读取 & 抽样
    print('1. 分块读取并抽样...')
    chunks = []
    total = 0
    for ch in tqdm(pd.read_csv(DATA_FILE, parse_dates=['timestamp','date'], chunksize=500_000), desc='Sampling'):
        total += len(ch)
        chunks.append(ch.sample(frac=SAMPLE_FRAC, random_state=42))
    df_sample = pd.concat(chunks, ignore_index=True)
    print(f'  读取 {total} 条, 抽样 {len(df_sample)} 条')
    if len(df_sample) > MAX_SAMPLES:
        df_sample = df_sample.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f'  限制样本至 {MAX_SAMPLES} 条')

    # 2. 标注 & 分数
    print('2. 标注 & 计算分数...')
    df_sample = load_true_bad(df_sample)
    df_sample = compute_scores(df_sample)
    df_sample = df_sample[['station_id','date','s5','s7','s8','true_bad']]

    # 3. 阶段1: 粗调
    print('3. 阶段1: 粗调搜索...')
    coarse = list(product(default_alphas(), default_alphas(), default_alphas(), default_thresholds()))
    res1 = Parallel(n_jobs=N_JOBS)(delayed(eval_one)(p, df_sample) for p in coarse)
    df1 = pd.DataFrame(res1)
    best1 = df1.loc[df1['F1_sum'].idxmax()]
    print('  粗调最优:', best1.to_dict())

    # 4. 阶段2: 细调
    print('4. 阶段2: 细调搜索...')
    a5, a7, a8, th = best1[['alpha5','alpha7','alpha8','threshold']]
    alphas = sorted({max(0, a5-0.5), a5, min(1, a5+0.5)})
    threshs = sorted({max(0, th-0.3), th, min(1, th+0.3)})
    fine = list(product(alphas, alphas, alphas, threshs))
    res2 = Parallel(n_jobs=N_JOBS)(delayed(eval_one)(p, df_sample) for p in fine)
    df2 = pd.DataFrame(res2)

    # 5. 保存 & 绘图
    df_tune = pd.concat([df1, df2], ignore_index=True)
    df_tune.to_csv(OUTPUT_TUNE, index=False)
    pivot = df_tune.pivot_table('F1_sum', index='alpha5', columns='threshold')
    plt.figure(figsize=(6,4))
    sns.heatmap(pivot, annot=True)
    plt.title('Alpha5 vs Threshold (sum F1)')
    plt.savefig(OUTPUT_PLOT)
    print('  调优结果 & 热力图已保存')

    # 6. 全量应用最优参数
    best = df2.loc[df2['F1_sum'].idxmax()]
    print('5. 最终最优参数:', best.to_dict())
    print('6. 分块应用最优掩膜...')
    first = True
    for ch in tqdm(pd.read_csv(DATA_FILE, parse_dates=['timestamp','date'], chunksize=500_000), desc='Apply Mask'):
        ch = load_true_bad(ch)
        ch = compute_scores(ch)
        ch['score'] = best['alpha5']*ch['s5'] + best['alpha7']*ch['s7'] + best['alpha8']*ch['s8']
        ch['mask_logic'] = (ch['score'] > best['threshold']).astype(int)
        ch.to_csv(OUTPUT_BEST, mode='w' if first else 'a', header=first, index=False)
        first = False
    print('  最终掩膜结果保存:', OUTPUT_BEST)

if __name__=='__main__':
    main()