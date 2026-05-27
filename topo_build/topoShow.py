import pandas as pd
import torch
import networkx as nx

# -- 1. note --
ei = torch.load("edge_index.pt")
G  = nx.DiGraph()
G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
comps = sorted(nx.weakly_connected_components(G), key=len)
small = comps[0]
print("note ID: ", sorted(small))

# -- 2. notedata --
meta_path = "/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt"
df = pd.read_csv(meta_path, sep='\t', engine='python')
df.columns = df.columns.str.strip()

# -- 3. note --
info_cols = ['ID','Fwy','Abs_PM','Name']
print("\n=== notedata ===")
print(df[df['ID'].isin(small)][info_cols])


""""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/topo_build/topoShow.py
note ID:  [775949, 775950, 775951, 775961, 775962, 775963, 775975, 775976]

=== notedata ===
          ID  Fwy  Abs_PM                  Name
4719  775949  126  39.478    COMMERCE CENTER DR
4720  775950  126  39.478    COMMERCE CENTER DR
4721  775951  126  39.478    COMMERCE CENTER DR
4722  775961  126  39.398    COMMERCE CENTER DR
4723  775962  126  39.398    COMMERCE CENTER DR
4724  775963  126  39.398    COMMERCE CENTER DR
4725  775975  126  39.398  COMMERCE CENTER DR.2
4726  775976  126  39.398  COMMERCE CENTER DR.2
"""