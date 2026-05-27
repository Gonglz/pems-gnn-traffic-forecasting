import pandas as pd

# 1. readnotedata(note)
meta_main_path = '/scratch/lgong1/finalproject/pems_data/pems_detector/step01_station_meta.csv'
meta_main = pd.read_csv(meta_main_path)

# 2. note
missing = meta_main[meta_main['latitude'].isnull() | meta_main['longitude'].isnull()]
print("notedatanote: ")
print(missing[['station_id', 'latitude', 'longitude']])


"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step01_metaCheck.py
notedatanote:
      station_id  latitude  longitude
1812      718156       NaN        NaN
2220      760349       NaN        NaN
2221      760350       NaN        NaN
2223      760361       NaN        NaN
3845      770172       NaN        NaN

process finished, exit codenote 0
"""
import pandas as pd

# 1. note station_id note
missing_ids = [718156, 760349, 760350, 760361, 770172]

# 2. note
mask = pd.read_csv(
    '/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv',
    parse_dates=['timestamp']
)

# 3. note
mask_missing = mask[mask['station_id'].isin(missing_ids)]

total = len(mask_missing)
print(f"note {len(missing_ids)} note, note {total} rows\n")

# 4. notedistribution
for col in ['mask_logic', 'mask_md', 'mask_hf']:
    cnt = mask_missing[col].astype(bool).sum()
    pct = cnt / total * 100
    print(f"{col:12s}: {cnt:10d} rows, {pct:6.2f}%")
"""note 5 note, note 165885 rows

mask_logic:          0 rows,   0.00%
mask_md:      17766 rows,  10.71%
mask_hf:      82425 rows,  49.69%

process finished, exit codenote 0"""