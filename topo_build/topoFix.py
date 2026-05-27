import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 1. note edge_index
ei = torch.load("edge_index.pt")
edges = set(zip(ei[0].tolist(), ei[1].tolist()))

# 2. readnotedata, note
df = pd.read_csv("/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt", sep='\t', engine='python')
df = df.rename(columns={'ID':'id','Latitude':'lat','Longitude':'lon'})
df = df.dropna(subset=['lat','lon'])
ids = df['id'].astype(int).tolist()
coords = df[['lat','lon']].astype(float).values

# 3. note small_comp note df note
small_comp = [775949,775950,775951,775961,775962,775963,775975,775976]
id_to_idx = {i:idx for idx,i in enumerate(ids)}

# 4. KNN note
nbrs = NearestNeighbors(n_neighbors=2, metric='haversine').fit(np.radians(coords))
neighs = nbrs.kneighbors(np.radians(coords), return_distance=False)

for node in small_comp:
    idx = id_to_idx[node]
    # neighs[idx][0] note, note
    neighbor = ids[neighs[idx][1]]
    edges.add((node, neighbor))
    edges.add((neighbor, node))  # note

# 5. savenote edge_index
new_ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
torch.save(new_ei, "edge_index_patched.pt")
print("Patched graph saved. note 1 note.")
