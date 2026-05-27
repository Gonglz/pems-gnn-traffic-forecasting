import torch
import pandas as pd

# 1. notegeneratenote edge_index
edge_index = torch.load("edge_index.pt")  # shape [2, E]
# note ID
nodes_in_graph = set(edge_index.numpy().flatten().tolist())

# 2. readnotedata, note
#    notedatanote tab note txt:
df = pd.read_csv(
    "/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt",
    sep="\t", engine="python"
)
all_nodes = set(df["ID"].astype(int).tolist())

# 3. computenote: notedatanote, note
missing_nodes = all_nodes - nodes_in_graph

print(f"note {len(all_nodes)} notedatanote, note {len(nodes_in_graph)} note.")
print(f"note {len(missing_nodes)} note: \n", sorted(missing_nodes))

# topoCheck_fixed.py

import torch
import networkx as nx

# 1. note edge_index
edge_index = torch.load("edge_index.pt")  # [2, E]
src_list = edge_index[0].tolist()
dst_list = edge_index[1].tolist()

# 2. note
G = nx.DiGraph()
G.add_edges_from(zip(src_list, dst_list))

# 3. note
components = list(nx.weakly_connected_components(G))
print("note: ", len(components))
sizes = sorted((len(c) for c in components), reverse=True)
print("note(first5): ", sizes[:5])

# 4. note
isolated = list(nx.isolates(G))
print("note: ", len(isolated))
if isolated:
    print("note: ", isolated[:20])

# 5. notedistributionnote
degrees = dict(G.degree())
deg_vals = list(degrees.values())
print("note: ", sum(deg_vals)/len(deg_vals))
print("note: ", max(deg_vals))
print("note1note: ", sum(1 for d in deg_vals if d==1))

import numpy as np

degrees = [d for _, d in G.degree()]
print("note: ", np.mean(degrees))
print("note: ", np.max(degrees))
print("note 1 note: ", sum(np.array(degrees)==1))


import torch, networkx as nx

edge_index = torch.load("edge_index.pt")
G = nx.DiGraph()
G.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

# note, note
comps = sorted(nx.weakly_connected_components(G), key=len)
small = comps[0]  # note
print("note: ", sorted(small))


# note1note
deg1 = [n for n,d in G.degree() if d==1]
print("note1note: ", deg1)

""""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/topo_build/topoCheck.py
note 4888 notedatanote, note 4888 note.
note 0 note:
 []
note:  2
note(first5):  [4880, 8]
note:  0
note:  11.01963993453355
note:  19
note1note:  3
note:  11.01963993453355
note:  19
note 1 note:  3
note:  [775949, 775950, 775951, 775961, 775962, 775963, 775975, 775976]
note1note:  [760361, 770172, 760349]

process finished, exit codenote 0"""