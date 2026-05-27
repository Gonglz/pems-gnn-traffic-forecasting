# topoBuild_tab_fixed.py
""""
  /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/topo_build/topoBuild.py
✔ 拓扑生成完毕，edge_index.pt shape = (2, 26932)
"""
import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_topology(meta_txt: str, knn_k: int = 3):
    df = pd.read_csv(meta_txt, sep='\t', engine='python')
    df.columns = [c.strip() for c in df.columns]

    # 重命名
    df = df.rename(columns={
        'ID': 'id',
        'Abs_PM': 'abs_pm',
        'Latitude': 'lat',
        'Longitude': 'lon'
    })
    df['id']     = df['id'].astype(int)
    df['abs_pm'] = df['abs_pm'].astype(float)

    # 主干连边
    edges = []
    for (_, d), grp in df.groupby(['Fwy','Dir']):
        grp = grp.sort_values('abs_pm')
        ids = grp['id'].tolist()
        edges += [(u, v) for u, v in zip(ids, ids[1:])]

    # KNN 辅边：先过滤掉缺失坐标的行
    if knn_k > 0:
        knn_df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
        coords = knn_df[['lat','lon']].astype(float).values
        # 转弧度后用 haversine
        nbrs   = NearestNeighbors(n_neighbors=knn_k+1,
                                  metric='haversine') \
                 .fit(np.radians(coords))
        neighs = nbrs.kneighbors(np.radians(coords),
                                  return_distance=False)
        ids_knn = knn_df['id'].tolist()
        for i, nbr in enumerate(neighs):
            src = ids_knn[i]
            for j in nbr[1:]:  # skip self
                dst = ids_knn[j]
                edges.append((src, dst))

    # 去重 & edge_index
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

if __name__ == "__main__":
    meta_path = "/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt"
    ei = build_topology(meta_path, knn_k=5)
    torch.save(ei, "edge_index.pt")
    print("✔ 拓扑生成完毕，edge_index.pt shape =", tuple(ei.shape))
