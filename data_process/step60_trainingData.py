#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step60_prepare_training_data.py

1. readnoteresultnotedata(note ID note station_id)
2. note, note timestamp x station_id note
3. note & 7 note Z-score note
4. Pivot note (T, N, F) note X note (T, N) note Y
5. save X.npy, Y.npy note PyG note data_list_*.pt(note torch_geometric)
"""

import os
import numpy as np
import pandas as pd
import torch

# configuration
INTERP_PARQ   = '/scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet'
META_CSV      = '/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv'
EDGE_INDEX_PT = '/scratch/lgong1/finalproject/pems_data/step52_edge_index.pt'
FEATURE_DIR   = '/scratch/lgong1/finalproject/pems_data'

# note P (12 note = 1h)
P             = 12
# note Δ
DELTAS        = {'5min':1, '15min':3, '30min':6}
# note
NORM_WINDOW   = 7 * 24 * 12
MIN_PERIODS   = 24 * 12  # notedata

def main():
    # 1. readnoteresult
    df = pd.read_parquet(INTERP_PARQ)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. readnotedatanote station_id note
    meta = pd.read_csv(META_CSV)
    if 'station_id' not in meta.columns and 'ID' in meta.columns:
        meta = meta.rename(columns={'ID':'station_id'})
    # note station_id
    meta = meta.drop_duplicates(subset=['station_id'])
    # notestaticnote(note)
    static_cols = []
    for c in ['lanes','Lanes']:
        if c in meta.columns: static_cols.append(c)
    for c in ['length','Length']:
        if c in meta.columns and c not in static_cols: static_cols.append(c)
    # note
    df = df.merge(meta[['station_id'] + static_cols],
                  on='station_id', how='left')

    # 3. note: note timestamp x station_id note
    df = df.drop_duplicates(subset=['timestamp','station_id'])

    # 4. note
    df['hour']     = df['timestamp'].dt.hour + df['timestamp'].dt.minute/60
    df['sin_hour'] = np.sin(2*np.pi * df['hour']/24)
    df['cos_hour'] = np.cos(2*np.pi * df['hour']/24)
    df['dow']      = df['timestamp'].dt.weekday
    df['sin_dow']  = np.sin(2*np.pi * df['dow']/7)
    df['cos_dow']  = np.cos(2*np.pi * df['dow']/7)

    # 5. note Z-score note
    ts_feats = ['flow_interp','occupancy_interp','speed_interp']
    for feat in ts_feats:
        col_n = f'{feat}_norm'
        df[col_n] = (
            df.groupby('station_id')[feat].transform(lambda x: (x - x.rolling(NORM_WINDOW, min_periods=MIN_PERIODS).mean())
                                  / x.rolling(NORM_WINDOW, min_periods=MIN_PERIODS).std())
        )

    # 6. note pivot firstnote DataFrame
    # note (timestamp, station_id)note, note
    # note X note
    norm_cols   = [f'{f}_norm' for f in ts_feats]
    time_cols   = ['sin_hour','cos_hour','sin_dow','cos_dow']
    feature_cols= norm_cols + time_cols + static_cols
    df_all = df.set_index(['timestamp','station_id'])[feature_cols + ['flow_interp']]

    # 7. note ts_list, st_list
    ts_list = sorted(df_all.index.get_level_values('timestamp').unique())
    st_list = sorted(df_all.index.get_level_values('station_id').unique())
    T, N = len(ts_list), len(st_list)
    F = len(feature_cols)

    # 8. note X, Y
    X = np.zeros((T, N, F), dtype=np.float32)
    Y = np.zeros((T, N),    dtype=np.float32)

    # 9. Pivot note Y (goalnote flow_interp note)
    df_y = df_all['flow_interp'].unstack(level='station_id')
    Y[:,:] = df_y.loc[ts_list, st_list].values

    # 10. Pivot note X
    for i, feat in enumerate(feature_cols):
        df_f = df_all[feat].unstack(level='station_id')
        X[:,:, i] = df_f.loc[ts_list, st_list].values

    # 11. save
    np.save(os.path.join(FEATURE_DIR,'X.npy'), X)
    np.save(os.path.join(FEATURE_DIR,'Y.npy'), Y)
    print(f"Saved X.npy {X.shape}, Y.npy {Y.shape}")

    # 12. (note) PyG DataList
    try:
        from torch_geometric.data import Data
        edge_index = torch.load(EDGE_INDEX_PT)
        for name, delta in DELTAS.items():
            T_eff = T - delta
            X_trim = X[:T_eff]
            Y_trim = Y[delta:delta+T_eff]
            lst = []
            for t in range(P-1, T_eff):
                xh = X_trim[t-P+1:t+1]
                yt = Y_trim[t]
                data = Data(x=torch.from_numpy(xh).float(),
                            y=torch.from_numpy(yt).float(),
                            edge_index=edge_index)
                lst.append(data)
            outp = os.path.join(FEATURE_DIR, f'data_list_{name}.pt')
            torch.save(lst, outp)
            print(f"Saved {len(lst)} samples to {outp}")
    except ImportError:
        print("torch_geometric not installed; skipped DataList.")

if __name__ == '__main__':
    main()
