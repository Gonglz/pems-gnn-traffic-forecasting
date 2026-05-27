#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step02_weather_history_grid.py

note:
 1. readnotedata(step01_d07_meta.csv), note
 2. note10kmx10kmnotegeneratenote
 3. note KDTree note(note grid_id)
 4. note grid_id, note Meteostat note Hourly notedata
 5. note 5 note
 6. note station_id
 7. outputnote 5 note parquet

note:
  python step02_weather_history_grid.py

note:
  pip install pandas numpy pytz meteostat scikit-learn tqdm pyarrow

  /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step01_weather.py
[WARN] note: [718156, 760349, 760350, 760361, 770172]
[INFO] note: 4883
[INFO] generate 220 note
[INFO] note 67 note
[INFO] note: 2025-01-01 -> 2025-04-27
[INFO] note (2025-01-01 -> 2025-04-27)
note: 100%|██████████| 220/220 [00:11<00:00, 19.06it/s]
[INFO] note: 6278636
[INFO] note: 162710195, note: 4883
[INFO] savenote /scratch/lgong1/finalproject/pems_data/weather_5min_history.parquet

process finished, exit codenote 0

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

# note
LA_TZ = pytz.timezone('America/Los_Angeles')

# pathconfiguration
BASE_DIR    = Path(__file__).resolve().parent.parent / 'pems_data'
META_CSV    = BASE_DIR / 'step01_d07_meta.csv'
OUT_PARQ    = BASE_DIR / 'weather_5min_history.parquet'

# note: 10 km
GRID_KM     = 10.0

def load_meta(path):
    df = pd.read_csv(path, usecols=['station_id','latitude','longitude'])
    bad = df[df[['latitude','longitude']].isnull().any(axis=1)]
    if not bad.empty:
        print(f"[WARN] note: {bad['station_id'].tolist()}")
        df = df.dropna(subset=['latitude','longitude'])
    print(f"[INFO] note: {len(df)}")
    return df


def make_grid(stations, km=GRID_KM):
    lat_min, lat_max = stations.latitude.min(), stations.latitude.max()
    lon_min, lon_max = stations.longitude.min(), stations.longitude.max()
    ddeg = km / 111.0  # 1°~111km
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
    print(f"[INFO] generate {len(grid)} note")
    return grid


def map_stations_to_grid(stations, grid):
    pts = stations[['latitude','longitude']].values
    tree = KDTree(grid[['latitude','longitude']].values)
    dist, idx = tree.query(pts, k=1)
    mapping = pd.DataFrame({
        'station_id': stations.station_id.values,
        'grid_id':    grid.iloc[idx.flatten()]['grid_id'].values
    })
    print(f"[INFO] note {mapping.grid_id.nunique()} note")
    return mapping


def fetch_and_resample(grid, start, end):
    all_dfs = []
    print(f"[INFO] note ({start.date()} -> {end.date()})")
    for _, row in tqdm(grid.iterrows(), total=len(grid), desc='note'):
        gid = row.grid_id
        lat, lon = row.latitude, row.longitude
        loc = Point(lat, lon)
        hr = Hourly(loc, start, end, timezone=LA_TZ.zone).fetch()
        if hr.empty:
            continue
        # note
        col_map = {
            'temp': 'tavg', 'prcp': 'pcpn',
            'rhum': 'humidity', 'pres': 'pressure', 'wspd': 'wind_speed'
        }
        hr = hr.rename(columns=col_map)
        # note
        hr.index = hr.index.tz_convert(LA_TZ)
        # note5note
        df5 = hr.resample('5T').interpolate().reset_index()
        # note timestamp
        idx_col = df5.columns[0]
        df5 = df5.rename(columns={idx_col: 'timestamp'})
        df5['grid_id'] = gid
        # note
        cols = ['timestamp','tavg','pcpn','humidity','pressure','wind_speed','grid_id']
        all_dfs.append(df5[cols])
    if not all_dfs:
        raise RuntimeError("notedata")
    return pd.concat(all_dfs, ignore_index=True)


def main():
    stations = load_meta(META_CSV)
    grid = make_grid(stations)
    mapping = map_stations_to_grid(stations, grid)
    start = pd.to_datetime('2025-01-01')
    end   = pd.to_datetime('2025-04-27')
    print(f"[INFO] note: {start.date()} -> {end.date()}")
    hist = fetch_and_resample(grid, start, end)
    print(f"[INFO] note: {len(hist)}")
    out = mapping.merge(hist, on='grid_id', how='left')
    print(f"[INFO] note: {len(out)}, note: {out.station_id.nunique()}")
    out.to_parquet(OUT_PARQ, index=False)
    print(f"[INFO] savenote {OUT_PARQ}")

if __name__ == '__main__':
    main()
