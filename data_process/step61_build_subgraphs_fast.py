#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step61_build_subgraphs_fast.py

dynamicnote, note, note30note: note
- notesavenote X.npy/Y.npy note
- note step52_buildTopo.py notedatanote edge_index
- BFS note"note"noterows, note station_id -> X/Y note
- note subgraph, note PyG subgraph(), note
- note & ETA note
"""

import os, time
import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm.auto import tqdm

# ─── configuration ───────────────────────────────────────────────────────────────
DATA_DIR         = '/scratch/lgong1/finalproject/pems_data'
X_PATH           = os.path.join(DATA_DIR, 'X.npy')
Y_PATH           = os.path.join(DATA_DIR, 'Y.npy')
INTERP_PARQ      = os.path.join(DATA_DIR, 'step51_interpolated_final.parquet')
META_CSV         = os.path.join(DATA_DIR, 'step01_d07_meta.csv')
EDGE_INDEX_PT    = os.path.join(DATA_DIR, 'step52_edge_index.pt')
OUTPUT_DATA_LIST = os.path.join(DATA_DIR, 'data_list_multihead_subgraph_fast.pt')

# note(1 note = 5 note)
DELTAS = {'5min': 1, '15min': 3, '30min': 6}
# note = note(1 mi/h) x note(note)
HOURS  = {name: delta * 5 / 60 for name, delta in DELTAS.items()}
# ───────────────────────────────────────────────────────────────────────────

def main():
    # 1. note/note
    print("Loading X.npy, Y.npy...", end="", flush=True)
    X = np.load(X_PATH)    # shape (T, N_all, F)
    Y = np.load(Y_PATH)    # shape (T, N_all)
    print(" done.")

    # 2. note step60 note station_id note
    print("Reconstructing station_id list...", end="", flush=True)
    df_ids = pd.read_parquet(INTERP_PARQ, columns=['station_id']).drop_duplicates()
    df_ids['station_id'] = df_ids['station_id'].astype(int)
    full_st_list = sorted(df_ids['station_id'].tolist())
    st2i = {s: i for i, s in enumerate(full_st_list)}
    print(f" done. total stations = {len(full_st_list)}")

    # 3. note & note meta(unifiednote)
    print("Loading edge_index and building graph...", end="", flush=True)
    edge_index = torch.load(EDGE_INDEX_PT)  # [2, E]
    rows, cols = edge_index.numpy()

    meta = pd.read_csv(META_CSV)
    meta.columns = [c.lower() for c in meta.columns]
    if 'id' in meta.columns and 'station_id' not in meta.columns:
        meta = meta.rename(columns={'id':'station_id'})
    # noterows
    meta = meta.dropna(subset=['latitude','longitude'])
    # note full_st_list note
    meta = meta[meta['station_id'].isin(full_st_list)].reset_index(drop=True)

    # graph_nodes[idx] = station_id
    graph_nodes = meta['station_id'].astype(int).tolist()
    length_map  = dict(zip(meta['station_id'], meta['length']))

    # note, note 0...len(graph_nodes)-1
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_nodes)))
    for u, v in zip(cols, rows):
        G.add_edge(int(u), int(v),
                   length=length_map.get(graph_nodes[int(u)], 1.0))
    print(" done.")

    # 4. note
    df_ts = pd.read_parquet(INTERP_PARQ, columns=['timestamp']).drop_duplicates()
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
    ts_list = sorted(df_ts['timestamp'].tolist())
    ts2i    = {t: i for i, t in enumerate(ts_list)}

    # 5. note
    total = len(ts_list) * len(graph_nodes)
    print(f"Sampling subgraphs for {total} (ts,node) pairs...")
    data_list = []
    count = 0
    t0 = time.time()

    edge_rows = rows
    edge_cols = cols

    for ts in tqdm(ts_list, desc='timestamps'):
        ti = ts2i[ts]
        for node_idx, sid in enumerate(graph_nodes):
            count += 1
            if count % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"[{count}/{total}] "
                      f"elapsed {elapsed:.1f}s, "
                      f"avg {elapsed/count*1000:.2f}ms/sample, "
                      f"ETA {eta/60:.1f}min")

            # 5a) note BFS in graph node indices
            sub_idx = {}
            for name, maxd in HOURS.items():
                visited = {node_idx}
                queue   = [(node_idx, 0.0)]
                while queue:
                    u, dist = queue.pop(0)
                    for v, attr in G[u].items():
                        nd = dist + attr['length']
                        if nd <= maxd and v not in visited:
                            visited.add(v)
                            queue.append((v, nd))
                sub_idx[name] = list(visited)

            # 5b) note, note, note
            xs, eis, ys = {}, {}, {}
            for name, delta in DELTAS.items():
                idxs = sub_idx[name]
                # map graph_nodes[idx] -> station_id -> col_idx in X/Y
                col_idxs = [st2i[ graph_nodes[i] ] for i in idxs]
                x_sub = X[ti, col_idxs,:]
                y_sub = Y[ti + delta, st2i[sid]]

                # note
                mask = np.isin(edge_rows, idxs) & np.isin(edge_cols, idxs)
                sub_rows = edge_rows[mask]
                sub_cols = edge_cols[mask]
                # local mapping old->new
                local_map = {old:i for i,old in enumerate(idxs)}
                new_rows = [ local_map[r] for r in sub_rows ]
                new_cols = [ local_map[c] for c in sub_cols ]
                new_ei = torch.tensor([new_rows, new_cols], dtype=torch.long)

                center_new_idx = local_map[node_idx]

                xs[name]  = torch.from_numpy(x_sub).float()
                eis[name] = new_ei
                ys[name]  = torch.tensor(y_sub, dtype=torch.float)

            data = Data(
                x5=  xs['5min'],  edge_index5=  eis['5min'],  y5=  ys['5min'],
                x15= xs['15min'],edge_index15= eis['15min'],y15= ys['15min'],
                x30= xs['30min'],edge_index30= eis['30min'],y30= ys['30min'],
                center_idx=torch.tensor(center_new_idx, dtype=torch.long)
            )
            data_list.append(data)

    # 6. save
    torch.save(data_list, OUTPUT_DATA_LIST)
    total_time = time.time() - t0
    print(f"Saved {len(data_list)} samples to {OUTPUT_DATA_LIST}")
    print(f"Total runtime: {total_time:.1f}s "
          f"({total_time/total*1000:.2f}ms per sample)")

if __name__ == '__main__':
    main()


""""[779000/162003291] elapsed 550.3s, avg 0.71ms/sample, ETA 1898.1min
[780000/162003291] elapsed 550.9s, avg 0.71ms/sample, ETA 1897.9min
[781000/162003291] elapsed 551.6s, avg 0.71ms/sample, ETA 1897.7min
timestamps:   0%|          | 160/33177 [09:11<30:30:49,  3.33s/it][782000/162003291] elapsed 552.2s, avg 0.71ms/sample, ETA 1897.5min
[783000/162003291] elapsed 552.9s, avg 0.71ms/sample, ETA 1897.5min
[784000/162003291] elapsed 553.7s, avg 0.71ms/sample, ETA 1897.6min
[785000/162003291] elapsed 554.3s, avg 0.71ms/sample, ETA 1897.4min
[786000/162003291] elapsed 555.0s, avg 0.71ms/sample, ETA 1897.2min
timestamps:   0%|          | 161/33177 [09:15<30:30:33,  3.33s/it][787000/162003291] elapsed 555.6s, avg 0.71ms/sample, ETA 1897.0min
[788000/162003291] elapsed 556.4s, avg 0.71ms/sample, ETA 1897.1min
[789000/162003291] elapsed 557.1s, avg 0.71ms/sample, ETA 1897.1min
[790000/162003291] elapsed 557.7s, avg 0.71ms/sample, ETA 1896.9min
[791000/162003291] elapsed 558.4s, avg 0.71ms/sample, ETA 1896.7min
timestamps:   0%|          | 162/33177 [09:18<30:29:13,  3.32s/it][792000/162003291] elapsed 559.0s, avg 0.71ms/sample, ETA 1896.5min
[793000/162003291] elapsed 559.8s, avg 0.71ms/sample, ETA 1896.6min
[794000/162003291] elapsed 560.5s, avg 0.71ms/sample, ETA 1896.6min
[795000/162003291] elapsed 561.1s, avg 0.71ms/sample, ETA 1896.4min
timestamps:   0%|          | 163/33177 [09:21<30:32:08,  3.33s/it][796000/162003291] elapsed 561.8s, avg 0.71ms/sample, ETA 1896.3min
[797000/162003291] elapsed 562.5s, avg 0.71ms/sample, ETA 1896.2min
[798000/162003291] elapsed 563.2s, avg 0.71ms/sample, ETA 1896.2min
[799000/162003291] elapsed 563.9s, avg 0.71ms/sample, ETA 1896.2min
[800000/162003291] elapsed 564.5s, avg 0.71ms/sample, ETA 1896.0min
timestamps:   0%|          | 164/33177 [09:25<30:33:33,  3.33s/it][801000/162003291] elapsed 565.2s, avg 0.71ms/sample, ETA 1895.9min"""