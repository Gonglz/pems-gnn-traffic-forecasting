import pandas as pd

# 1. 读取主站点元数据（包含大多数坐标）
meta_main_path = '/scratch/lgong1/finalproject/pems_data/pems_detector/step01_station_meta.csv'
meta_main = pd.read_csv(meta_main_path)

# 2. 检查缺失经纬度的站点
missing = meta_main[meta_main['latitude'].isnull() | meta_main['longitude'].isnull()]
print("主元数据中缺失坐标的站点：")
print(missing[['station_id', 'latitude', 'longitude']])


"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step01_metaCheck.py 
主元数据中缺失坐标的站点：
      station_id  latitude  longitude
1812      718156       NaN        NaN
2220      760349       NaN        NaN
2221      760350       NaN        NaN
2223      760361       NaN        NaN
3845      770172       NaN        NaN

进程已结束，退出代码为 0
"""
import pandas as pd

# 1. 缺失坐标的 station_id 列表
missing_ids = [718156, 760349, 760350, 760361, 770172]

# 2. 读入掩码表
mask = pd.read_csv(
    '/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv',
    parse_dates=['timestamp']
)

# 3. 筛出缺失坐标站点的记录
mask_missing = mask[mask['station_id'].isin(missing_ids)]

total = len(mask_missing)
print(f"针对这 {len(missing_ids)} 个站点，总计记录 {total} 行\n")

# 4. 统计三种掩码策略的分布
for col in ['mask_logic', 'mask_md', 'mask_hf']:
    cnt = mask_missing[col].astype(bool).sum()
    pct = cnt / total * 100
    print(f"{col:12s}: {cnt:10d} 行，{pct:6.2f}%")
"""针对这 5 个站点，总计记录 165885 行

mask_logic  :          0 行，  0.00%
mask_md     :      17766 行， 10.71%
mask_hf     :      82425 行， 49.69%

进程已结束，退出代码为 0"""