import pandas as pd
import torch

# 1. 读取站点元数据
meta_path = '/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv'
meta = pd.read_csv(meta_path, usecols=['station_id','latitude','longitude'])

# 2. 加载 edge_index_patched.pt，看里面包含哪些字段
pt_path = '/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv'
data = torch.load(pt_path)
print("PT 文件包含字段：", list(data.keys()))

# 3. 查看可能的坐标字段
#  假设有 'pos' 或 'x' 存储坐标
if hasattr(data, 'pos'):
    coords = data.pos.numpy()
    print("Found 'pos' with shape:", coords.shape)
elif 'pos' in data:
    coords = data['pos'].numpy()
    print("Found 'pos' with shape:", coords.shape)
elif 'x' in data:
    coords = data['x'].numpy()
    print("Found 'x' with shape:", coords.shape)
else:
    raise KeyError("无法在 PT 文件中找到坐标字段，请检查数据格式")

# 4. 假设 data.node_id 存储站点 ID 对应索引
#    如果没有，需要打印 data.node_id 或相关映射表
if hasattr(data, 'node_id'):
    node_ids = data.node_id.numpy()
else:
    # 尝试从其他键猜测
    print("请检查 PT 文件中的节点 ID 映射字段")

# 5. 构建坐标 DataFrame，并与元数据合并
coord_df = pd.DataFrame({
    'station_id': node_ids,
    'lat_pt': coords[:,0],
    'lon_pt': coords[:,1]
})
# 将缺失坐标用 PT 文件中的填充
meta_updated = meta.merge(coord_df, on='station_id', how='left')
meta_updated['latitude']  = meta_updated['latitude'].fillna(meta_updated['lat_pt'])
meta_updated['longitude'] = meta_updated['longitude'].fillna(meta_updated['lon_pt'])
meta_updated = meta_updated.drop(columns=['lat_pt','lon_pt'])

# 6. 保存新的元数据
meta_updated.to_csv(meta_path.replace('.csv','_filled.csv'), index=False)
print("已保存修复后元数据到:", meta_path.replace('.csv','_filled.csv'))
