#!/usr/bin/env python3
# coding: utf-8
"""
step31_fillExter.py

功能：
 1. 加载原始交通长表(step1_raw_long.csv)、历史 5 分钟粒度天气表(weather_5min_history.parquet)
    以及自定义事件表(custom_events.csv)
 2. 将所有 timestamp 列统一为无时区 datetime64[ns]
 3. 合并流量与天气数据
 4. 标记周末(is_weekend)、节假日(is_holiday)和自定义事件影响(in_custom_event)
 5. 输出包含所有外部特征的完整表(step31_fillExter.csv)

用法：
  cd finalproject/data_process
  python step31_fillExter.py

依赖：
  pip install pandas numpy pytz holidays tqdm pyarrow
  /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step31_fiillExter.py
加载并规范化原始流量数据…
加载并规范化历史天气数据…
合并流量与天气…
标记周末与节假日…
加载站点经纬度…
标记自定义事件影响…
事件循环: 100%|██████████| 12/12 [00:29<00:00,  2.43s/it]
写入最终结果…
✔ 完成：/scratch/lgong1/finalproject/pems_data/step31_fillExter.csv

进程已结束，退出代码为 0
"""
import os
import pandas as pd
import numpy as np
import pytz
import holidays
from tqdm import tqdm

# 配置
LA_TZ       = pytz.timezone('America/Los_Angeles')
CA_HOLIDAYS = holidays.CountryHoliday('US', prov='CA')
RADIUS_KM   = 1.0

# 路径
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_CSV       = os.path.join(BASE_DIR, 'pems_data', 'step1_raw_long.csv')
WEATHER_PARQ  = os.path.join(BASE_DIR, 'pems_data', 'weather_5min_history.parquet')
EVENTS_CSV    = os.path.join(BASE_DIR, 'pems_data', 'enrich', 'custom_events.csv')
OUT_CSV       = os.path.join(BASE_DIR, 'pems_data', 'step31_fillExter.csv')


def haversine(lon1, lat1, lon2, lat2):
    """计算两组经纬度点的 Haversine 距离（公里）"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def main():
    # 1. 读流量长表并去掉时区
    print('加载并规范化原始流量数据…')
    raw = pd.read_csv(RAW_CSV, parse_dates=['timestamp'])
    raw['timestamp'] = pd.to_datetime(raw['timestamp']).dt.tz_localize(None)

    # 2. 读历史天气并去掉时区（如果有）
    print('加载并规范化历史天气数据…')
    weather = pd.read_parquet(WEATHER_PARQ)
    # 统一成无时区
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    if hasattr(weather['timestamp'].dt, 'tz'):
        try:
            weather['timestamp'] = weather['timestamp'].dt.tz_convert(LA_TZ)
        except Exception:
            pass
    weather['timestamp'] = weather['timestamp'].dt.tz_localize(None)

    # 3. 合并流量与天气
    print('合并流量与天气…')
    df = raw.merge(
        weather,
        on=['station_id', 'timestamp'],
        how='left'
    )

    # 4. 标记周末 & 节假日
    print('标记周末与节假日…')
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    df['is_holiday'] = df['timestamp'].dt.date.isin(CA_HOLIDAYS)

    # 5. 读站点经纬度，用于事件影响计算
    print('加载站点经纬度…')
    meta = pd.read_csv(
        os.path.join(BASE_DIR, 'pems_data', 'step01_d07_meta.csv'),
        usecols=['station_id', 'latitude', 'longitude']
    ).dropna(subset=['latitude', 'longitude'])
    df = df.merge(meta, on='station_id', how='left')

    # 6. 标记自定义事件影响
    print('标记自定义事件影响…')
    df['in_custom_event'] = False
    lons = df['longitude'].values
    lats = df['latitude'].values
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_time','end_time'])
    for _, ev in tqdm(events.iterrows(), total=len(events), desc='事件循环'):
        mask_time = (df['timestamp'] >= ev['start_time']) & (df['timestamp'] <= ev['end_time'])
        if not mask_time.any():
            continue
        dist = haversine(
            lons[mask_time], lats[mask_time],
            ev['longitude'], ev['latitude']
        )
        idx = df.index[mask_time][dist <= RADIUS_KM]
        df.loc[idx, 'in_custom_event'] = True

    # 7. 输出
    print('写入最终结果…')
    df.to_csv(OUT_CSV, index=False)
    print(f'✔ 完成：{OUT_CSV}')


if __name__ == '__main__':
    main()
