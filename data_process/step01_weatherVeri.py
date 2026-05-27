"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step01_weatherVeri.py
Columns: ['station_id', 'grid_id', 'timestamp', 'tavg', 'pcpn', 'humidity', 'pressure', 'wind_speed']
Total records: 162710195
Unique stations: 4883
Unique grid cells: 67

Stations without weather mapping (5): [718156, 760349, 760350, 760361, 770172]...

Missing tavg: 11 / 162710195 (0.00%)
Missing pcpn: 11 / 162710195 (0.00%)
Missing humidity: 11 / 162710195 (0.00%)
Missing pressure: 11 / 162710195 (0.00%)
Missing wind_speed: 11 / 162710195 (0.00%)


process finished, exit codenote 0
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load
df = pd.read_parquet('/scratch/lgong1/finalproject/pems_data/weather_5min_history.parquet')

# 2. Inspect columns & basic shape
print("Columns:", df.columns.tolist())
print("Total records:", len(df))
print("Unique stations:", df['station_id'].nunique())
print("Unique grid cells:", df['grid_id'].nunique(), "\n")

# 3. Coverage: any stations without a grid?
stations = pd.read_csv('/scratch/lgong1/finalproject/pems_data/step01_d07_meta.csv', usecols=['station_id'])
missing = set(stations['station_id']) - set(df['station_id'].unique())
print(f"Stations without weather mapping ({len(missing)}):", sorted(list(missing))[:10], "...\n")

# 4. Missing values
for col in ['tavg','pcpn','humidity','pressure','wind_speed']:
    cnt = df[col].isna().sum()
    print(f"Missing {col}: {cnt} / {len(df)} ({cnt/len(df):.2%})")
print()

# 5. Distributions
for col in ['tavg','pcpn','humidity','pressure','wind_speed']:
    plt.figure(figsize=(5,3))
    df[col].dropna().hist(bins=50)
    plt.title(f"{col} distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()

plt.show()

df_long = pd.read_csv('/scratch/lgong1/finalproject/pems_data/step1_raw_long.csv', parse_dates=['timestamp'])
weather = pd.read_parquet('current_weather.parquet')
df = df_long.merge(weather, on=['grid_id','timestamp'], how='left')
for col in ['tavg','pcpn','humidity','pressure','wind_speed']:
    print(col, df[col].isna().mean())
