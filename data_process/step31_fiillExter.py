#!/usr/bin/env python3
# coding: utf-8
"""
step31_fillExter.py

note:
 1. note(step1_raw_long.csv), note 5 note(weather_5min_history.parquet)
    note(custom_events.csv)
 2. note timestamp noteunifiednote datetime64[ns]
 3. notedata
 4. note(is_weekend), note(is_holiday)note(in_custom_event)
 5. outputnote(step31_fillExter.csv)

note:
  cd finalproject/data_process
  python step31_fillExter.py

note:
  pip install pandas numpy pytz holidays tqdm pyarrow
  /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step31_fiillExter.py
notedata…
notedata…
note…
note…
note…
note…
note: 100%|██████████| 12/12 [00:29<00:00,  2.43s/it]
noteresult…
✔ note: /scratch/lgong1/finalproject/pems_data/step31_fillExter.csv

process finished, exit codenote 0
"""
import os
import pandas as pd
import numpy as np
import pytz
import holidays
from tqdm import tqdm

# configuration
LA_TZ       = pytz.timezone('America/Los_Angeles')
CA_HOLIDAYS = holidays.CountryHoliday('US', prov='CA')
RADIUS_KM   = 1.0

# path
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_CSV       = os.path.join(BASE_DIR, 'pems_data', 'step1_raw_long.csv')
WEATHER_PARQ  = os.path.join(BASE_DIR, 'pems_data', 'weather_5min_history.parquet')
EVENTS_CSV    = os.path.join(BASE_DIR, 'pems_data', 'enrich', 'custom_events.csv')
OUT_CSV       = os.path.join(BASE_DIR, 'pems_data', 'step31_fillExter.csv')


def haversine(lon1, lat1, lon2, lat2):
    """computenote Haversine note(note)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def main():
    # 1. note
    print('notedata…')
    raw = pd.read_csv(RAW_CSV, parse_dates=['timestamp'])
    raw['timestamp'] = pd.to_datetime(raw['timestamp']).dt.tz_localize(None)

    # 2. note(note)
    print('notedata…')
    weather = pd.read_parquet(WEATHER_PARQ)
    # unifiednote
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    if hasattr(weather['timestamp'].dt, 'tz'):
        try:
            weather['timestamp'] = weather['timestamp'].dt.tz_convert(LA_TZ)
        except Exception:
            pass
    weather['timestamp'] = weather['timestamp'].dt.tz_localize(None)

    # 3. note
    print('note…')
    df = raw.merge(
        weather,
        on=['station_id', 'timestamp'],
        how='left'
    )

    # 4. note & note
    print('note…')
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    df['is_holiday'] = df['timestamp'].dt.date.isin(CA_HOLIDAYS)

    # 5. note, notecompute
    print('note…')
    meta = pd.read_csv(
        os.path.join(BASE_DIR, 'pems_data', 'step01_d07_meta.csv'),
        usecols=['station_id', 'latitude', 'longitude']
    ).dropna(subset=['latitude', 'longitude'])
    df = df.merge(meta, on='station_id', how='left')

    # 6. note
    print('note…')
    df['in_custom_event'] = False
    lons = df['longitude'].values
    lats = df['latitude'].values
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_time','end_time'])
    for _, ev in tqdm(events.iterrows(), total=len(events), desc='note'):
        mask_time = (df['timestamp'] >= ev['start_time']) & (df['timestamp'] <= ev['end_time'])
        if not mask_time.any():
            continue
        dist = haversine(
            lons[mask_time], lats[mask_time],
            ev['longitude'], ev['latitude']
        )
        idx = df.index[mask_time][dist <= RADIUS_KM]
        df.loc[idx, 'in_custom_event'] = True

    # 7. output
    print('noteresult…')
    df.to_csv(OUT_CSV, index=False)
    print(f'✔ note: {OUT_CSV}')


if __name__ == '__main__':
    main()
