#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step02_weather_history_grid.py

功能：
 1. 读取交通站点元数据(step01_d07_meta.csv)，去掉无坐标点
 2. 在研究区域按10km×10km网格生成代表点
 3. 使用 KDTree 将每个站点映射到最近网格中心（记录 grid_id）
 4. 对每个 grid_id，调用 Meteostat 的 Hourly 接口拉取历史小时数据
 5. 重采样到 5 分钟粒度并做线性插值
 6. 将插值后的记录广播回每个 station_id
 7. 输出全量历史 5 分钟级天气 parquet

用法：
  python step02_weather_history_grid.py

依赖：
  pip install pandas numpy pytz meteostat scikit-learn tqdm pyarrow

  /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step01_weather.py
[WARN] 删除无坐标站点: [718156, 760349, 760350, 760361, 770172]
[INFO] 有效站点数: 4883
[INFO] 生成 220 个网格中心点
[INFO] 站点映射到 67 个网格中心
[INFO] 时间范围: 2025-01-01 → 2025-04-27
[INFO] 开始拉取历史天气 (2025-01-01 → 2025-04-27)
网格循环: 100%|██████████| 220/220 [00:11<00:00, 19.06it/s]
[INFO] 插值后记录数: 6278636
[INFO] 最终记录数: 162710195, 覆盖站点数: 4883
[INFO] 保存至 /scratch/lgong1/finalproject/pems_data/weather_5min_history.parquet

进程已结束，退出代码为 0

"""
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm
from meteostat import Hourly, Point
from sklearn.neighbors import KDTree

# 时区
LA_TZ = pytz.timezone('America/Los_Angeles')

# 路径配置
BASE_DIR    = Path(__file__).resolve().parent.parent / 'pems_data'
META_CSV    = BASE_DIR / 'step01_d07_meta.csv'
OUT_PARQ    = BASE_DIR / 'weather_5min_history.parquet'

# 网格大小：10 km
GRID_KM     = 10.0

def load_meta(path):
    df = pd.read_csv(path, usecols=['station_id','latitude','longitude'])
    bad = df[df[['latitude','longitude']].isnull().any(axis=1)]
    if not bad.empty:
        print(f"[WARN] 删除无坐标站点: {bad['station_id'].tolist()}")
        df = df.dropna(subset=['latitude','longitude'])
    print(f"[INFO] 有效站点数: {len(df)}")
    return df


def make_grid(stations, km=GRID_KM):
    lat_min, lat_max = stations.latitude.min(), stations.latitude.max()
    lon_min, lon_max = stations.longitude.min(), stations.longitude.max()
    ddeg = km / 111.0  # 1°≈111km
    lats = np.arange(lat_min, lat_max + ddeg, ddeg)
    lons = np.arange(lon_min, lon_max + ddeg, ddeg)
    centers, gids = [], []
    gid = 0
    for i in range(len(lats)-1):
        for j in range(len(lons)-1):
            centers.append((lats[i] + ddeg/2, lons[j] + ddeg/2))
            gids.append(f"grid_{gid}")
            gid += 1
    grid = pd.DataFrame(centers, columns=['latitude','longitude'])
    grid['grid_id'] = gids
    print(f"[INFO] 生成 {len(grid)} 个网格中心点")
    return grid


def map_stations_to_grid(stations, grid):
    pts = stations[['latitude','longitude']].values
    tree = KDTree(grid[['latitude','longitude']].values)
    dist, idx = tree.query(pts, k=1)
    mapping = pd.DataFrame({
        'station_id': stations.station_id.values,
        'grid_id':    grid.iloc[idx.flatten()]['grid_id'].values
    })
    print(f"[INFO] 站点映射到 {mapping.grid_id.nunique()} 个网格中心")
    return mapping


def fetch_and_resample(grid, start, end):
    all_dfs = []
    print(f"[INFO] 开始拉取历史天气 ({start.date()} → {end.date()})")
    for _, row in tqdm(grid.iterrows(), total=len(grid), desc='网格循环'):
        gid = row.grid_id
        lat, lon = row.latitude, row.longitude
        loc = Point(lat, lon)
        hr = Hourly(loc, start, end, timezone=LA_TZ.zone).fetch()
        if hr.empty:
            continue
        # 重命名并选取字段
        col_map = {
            'temp': 'tavg', 'prcp': 'pcpn',
            'rhum': 'humidity', 'pres': 'pressure', 'wspd': 'wind_speed'
        }
        hr = hr.rename(columns=col_map)
        # 转换到本地时区
        hr.index = hr.index.tz_convert(LA_TZ)
        # 重采样到5分钟并插值
        df5 = hr.resample('5T').interpolate().reset_index()
        # 重命名索引列为 timestamp
        idx_col = df5.columns[0]
        df5 = df5.rename(columns={idx_col: 'timestamp'})
        df5['grid_id'] = gid
        # 保留列顺序
        cols = ['timestamp','tavg','pcpn','humidity','pressure','wind_speed','grid_id']
        all_dfs.append(df5[cols])
    if not all_dfs:
        raise RuntimeError("未拉取到任何历史天气数据")
    return pd.concat(all_dfs, ignore_index=True)


def main():
    stations = load_meta(META_CSV)
    grid = make_grid(stations)
    mapping = map_stations_to_grid(stations, grid)
    start = pd.to_datetime('2025-01-01')
    end   = pd.to_datetime('2025-04-27')
    print(f"[INFO] 时间范围: {start.date()} → {end.date()}")
    hist = fetch_and_resample(grid, start, end)
    print(f"[INFO] 插值后记录数: {len(hist)}")
    out = mapping.merge(hist, on='grid_id', how='left')
    print(f"[INFO] 最终记录数: {len(out)}, 覆盖站点数: {out.station_id.nunique()}")
    out.to_parquet(OUT_PARQ, index=False)
    print(f"[INFO] 保存至 {OUT_PARQ}")

if __name__ == '__main__':
    main()
