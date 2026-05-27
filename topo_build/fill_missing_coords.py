import pandas as pd
import torch

# 1. readnotedata
meta_path = '/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv'
meta = pd.read_csv(meta_path, usecols=['station_id','latitude','longitude'])

# 2. note edge_index_patched.pt, note
pt_path = '/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv'
data = torch.load(pt_path)
print("PT filenote: ", list(data.keys()))

# 3. note
#  note 'pos' note 'x' note
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
    raise KeyError("note PT filenote, notedatanote")

# 4. note data.node_id note ID note
#    note, note data.node_id note
if hasattr(data, 'node_id'):
    node_ids = data.node_id.numpy()
else:
    # note
    print("note PT filenote ID note")

# 5. note DataFrame, notedatanote
coord_df = pd.DataFrame({
    'station_id': node_ids,
    'lat_pt': coords[:,0],
    'lon_pt': coords[:,1]
})
# note PT filenote
meta_updated = meta.merge(coord_df, on='station_id', how='left')
meta_updated['latitude']  = meta_updated['latitude'].fillna(meta_updated['lat_pt'])
meta_updated['longitude'] = meta_updated['longitude'].fillna(meta_updated['lon_pt'])
meta_updated = meta_updated.drop(columns=['lat_pt','lon_pt'])

# 6. savenotedata
meta_updated.to_csv(meta_path.replace('.csv','_filled.csv'), index=False)
print("notesavenotedatanote:", meta_path.replace('.csv','_filled.csv'))
