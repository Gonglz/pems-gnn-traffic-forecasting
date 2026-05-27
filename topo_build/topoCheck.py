import torch
import pandas as pd

# 1. 载入你生成的 edge_index
edge_index = torch.load("edge_index.pt")  # shape [2, E]
# 提取所有出现在图中的节点 ID
nodes_in_graph = set(edge_index.numpy().flatten().tolist())

# 2. 读取元数据，拿到完整的站点列表
#    如果你的元数据是用 tab 分隔的 txt：
df = pd.read_csv(
    "/scratch/lgong1/finalproject/pems_data/pems_detector/d07_text_meta_2023_12_22.txt",
    sep="\t", engine="python"
)
all_nodes = set(df["ID"].astype(int).tolist())

# 3. 计算差集：在元数据里有，但不在图中的节点
missing_nodes = all_nodes - nodes_in_graph

print(f"总共 {len(all_nodes)} 个元数据站点，图里有 {len(nodes_in_graph)} 个节点。")
print(f"有 {len(missing_nodes)} 个站点没有连入图：\n", sorted(missing_nodes))

# topoCheck_fixed.py

import torch
import networkx as nx

# 1. 载入 edge_index
edge_index = torch.load("edge_index.pt")  # [2, E]
src_list = edge_index[0].tolist()
dst_list = edge_index[1].tolist()

# 2. 构建有向图
G = nx.DiGraph()
G.add_edges_from(zip(src_list, dst_list))

# 3. 弱连通分量检查
components = list(nx.weakly_connected_components(G))
print("连通分量数量：", len(components))
sizes = sorted((len(c) for c in components), reverse=True)
print("各分量大小（前5）：", sizes[:5])

# 4. 孤立节点检查
isolated = list(nx.isolates(G))
print("孤立节点数量：", len(isolated))
if isolated:
    print("示例孤立节点：", isolated[:20])

# 5. 度分布统计
degrees = dict(G.degree())
deg_vals = list(degrees.values())
print("平均度：", sum(deg_vals)/len(deg_vals))
print("最大度：", max(deg_vals))
print("度为1的节点数：", sum(1 for d in deg_vals if d==1))

import numpy as np

degrees = [d for _, d in G.degree()]
print("平均度：", np.mean(degrees))
print("最大度：", np.max(degrees))
print("度为 1 的节点数：", sum(np.array(degrees)==1))


import torch, networkx as nx

edge_index = torch.load("edge_index.pt")
G = nx.DiGraph()
G.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

# 取所有连通分量，找出那个小的
comps = sorted(nx.weakly_connected_components(G), key=len)
small = comps[0]  # 最小的那个分量
print("小分量节点：", sorted(small))


# 列出度为1的节点
deg1 = [n for n,d in G.degree() if d==1]
print("度为1的节点：", deg1)

""""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/topo_build/topoCheck.py 
总共 4888 个元数据站点，图里有 4888 个节点。
有 0 个站点没有连入图：
 []
连通分量数量： 2
各分量大小（前5）： [4880, 8]
孤立节点数量： 0
平均度： 11.01963993453355
最大度： 19
度为1的节点数： 3
平均度： 11.01963993453355
最大度： 19
度为 1 的节点数： 3
小分量节点： [775949, 775950, 775951, 775961, 775962, 775963, 775975, 775976]
度为1的节点： [760361, 770172, 760349]

进程已结束，退出代码为 0"""