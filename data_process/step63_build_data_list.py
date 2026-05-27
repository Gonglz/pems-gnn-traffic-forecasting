#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step63_build_data_list_parallel_fixed.py

noterowsnote(note KeyError): note joblib note
- note X.npy, Y.npy, ts_list, full_st_list, edge_index
- note step62_neighbors.pkl
- note, note
"""

import os, time
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# ─── configuration ───────────────────────────────────────────────────────────────
DATA_DIR         = '/scratch/lgong1/finalproject/pems_data'
X_PATH           = os.path.join(DATA_DIR, 'X.npy')
Y_PATH           = os.path.join(DATA_DIR, 'Y.npy')
INTERP_PARQ      = os.path.join(DATA_DIR, 'step51_interpolated_final.parquet')
NEIGHBORS_PKL    = os.path.join(DATA_DIR, 'step62_neighbors.pkl')
EDGE_INDEX_PT    = os.path.join(DATA_DIR, 'step52_edge_index.pt')
OUTPUT_DATA_LIST = os.path.join(DATA_DIR, 'data_list_multihead_subgraph_parallel_fixed.pkl')

DELTAS = {'5min':1, '15min':3, '30min':6}

def build_for_ts(ts, ts2i, graph_nodes, neighbors, X, Y, edge_rows, edge_cols, st2i):
    ti = ts2i[ts]
    sublist = []
    for node_idx, sid in enumerate(graph_nodes):
        xs, eis, ys = {}, {}, {}
        for name, delta in DELTAS.items():
            nbrs = neighbors[name][node_idx]
            # note local_map notecurrentnote
            local_map = {old:i for i, old in enumerate(nbrs)}
            # note
            col_idxs = [st2i[graph_nodes[i]] for i in nbrs]
            x_sub = X[ti, col_idxs,:]
            y_sub = Y[ti + delta, st2i[sid]]
            # note
            mask = np.isin(edge_rows, nbrs) & np.isin(edge_cols, nbrs)
            sub_rows = edge_rows[mask]; sub_cols = edge_cols[mask]
            new_rows = [local_map[r] for r in sub_rows]
            new_cols = [local_map[c] for c in sub_cols]
            new_ei   = torch.tensor([new_rows, new_cols], dtype=torch.long)
            xs[name]  = torch.from_numpy(x_sub).float()
            eis[name] = new_ei
            ys[name]  = torch.tensor(y_sub, dtype=torch.float)
        center_idx = torch.tensor(local_map[node_idx], dtype=torch.long)
        data = Data(
            x5=xs['5min'], edge_index5=eis['5min'], y5=ys['5min'],
            x15=xs['15min'], edge_index15=eis['15min'], y15=ys['15min'],
            x30=xs['30min'], edge_index30=eis['30min'], y30=ys['30min'],
            center_idx=center_idx
        )
        sublist.append(data)
    return sublist

def main():
    # 1. load X, Y
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    # 2. build ts_list & full_st_list
    df_id = pd.read_parquet(INTERP_PARQ, columns=['timestamp','station_id']).drop_duplicates()
    df_id['timestamp']  = pd.to_datetime(df_id['timestamp'])
    df_id['station_id'] = df_id['station_id'].astype(int)
    ts_list_all      = sorted(df_id['timestamp'].unique())
    full_st_list     = sorted(df_id['station_id'].unique())
    ts2i             = {t:i for i,t in enumerate(ts_list_all)}
    st2i             = {s:i for i,s in enumerate(full_st_list)}
    max_delta        = max(DELTAS.values())
    ts_list          = ts_list_all[:-max_delta]
    # 3. load neighbors
    with open(NEIGHBORS_PKL,'rb') as f:
        pdict = pickle.load(f)
    graph_nodes = pdict['graph_nodes']
    neighbors   = pdict['neighbors']
    # 4. load edge_index once
    edge_index = torch.load(EDGE_INDEX_PT)
    edge_rows, edge_cols = edge_index.numpy()
    # 5. parallel build
    total = len(ts_list) * len(graph_nodes)
    print(f"Parallel building for {total} samples...")
    t0 = time.time()
    results = Parallel(n_jobs=8)(
        delayed(build_for_ts)(
            ts, ts2i, graph_nodes, neighbors, X, Y,
            edge_rows, edge_cols, st2i
        ) for ts in tqdm(ts_list, desc='parallel ts')
    )
    data_list = [d for sub in results for d in sub]
    torch.save(data_list, OUTPUT_DATA_LIST)
    print(f"Saved {len(data_list)} samples in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step63_build_data_list.py
Parallel building for 161973993 samples...
parallel ts:   0%|          | 152/33171 [08:14<30:40:01,  3.34s/it]"""