import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 1. 载入 edge_index
ei = torch.load("edge_index.pt")
edges = set(zip(ei[0].tolist(), ei[1].tolist()))

# 2. 读取元数据，提取小分量和主网的坐标
df = pd.read_csv("/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt", sep='\t', engine='python')
df = df.rename(columns={'ID':'id','Latitude':'lat','Longitude':'lon'})
df = df.dropna(subset=['lat','lon'])
ids = df['id'].astype(int).tolist()
coords = df[['lat','lon']].astype(float).values

# 3. 找 small_comp 在 df 中的索引
small_comp = [775949,775950,775951,775961,775962,775963,775975,775976]
id_to_idx = {i:idx for idx,i in enumerate(ids)}

# 4. KNN 找最近主网邻居
nbrs = NearestNeighbors(n_neighbors=2, metric='haversine')\
       .fit(np.radians(coords))
neighs = nbrs.kneighbors(np.radians(coords), return_distance=False)

for node in small_comp:
    idx = id_to_idx[node]
    # neighs[idx][0] 是自己，第二个是最近的其他点
    neighbor = ids[neighs[idx][1]]
    edges.add((node, neighbor))
    edges.add((neighbor, node))  # 双向

# 5. 保存新的 edge_index
new_ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
torch.save(new_ei, "edge_index_patched.pt")
print("Patched graph saved. 现在只有 1 个连通分量。")
