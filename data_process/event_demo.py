""""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/event_demo.py
=== first 5 rowsdata ===
   timestamp  station_id  freeway_id... is_weekend is_holiday  in_custom_event
0 2025-01-01      715898           5...      False      False            False
1 2025-01-01      715900           5...      False      False            False
2 2025-01-01      715901           5...      False      False            False
3 2025-01-01      715903           5...      False      False            False
4 2025-01-01      715904           5...      False      False            False

[5 rows x 19 columns]

=== datanote ===
(162169176, 19)

=== field types ===
timestamp          datetime64[ns]
station_id                  int64
freeway_id                  int64
direction                  object
lane_type                  object
station_length            float64
samples                     int64
pct_observed                int64
flow                      float64
occupancy                 float64
speed                     float64
latitude                  float64
longitude                 float64
met_station                object
tavg                      float64
pcpn                      float64
is_weekend                   bool
is_holiday                   bool
in_custom_event              bool
dtype: object

=== missing valuesnote ===
                 missing_count  missing_pct
timestamp                    0     0.000000
station_id                   0     0.000000
freeway_id                   0     0.000000
direction                    0     0.000000
lane_type                    0     0.000000
station_length        69638523    42.941899
samples                      0     0.000000
pct_observed                 0     0.000000
flow                  53003095    32.683828
occupancy             53003095    32.683828
speed                 69638523    42.941899
latitude                165885     0.102291
longitude               165885     0.102291
met_station                  0     0.000000
tavg                   8659197     5.339607
pcpn                   8659197     5.339607
is_weekend                   0     0.000000
is_holiday                   0     0.000000
in_custom_event              0     0.000000

=== is_weekend distribution ===
False    116842752
True      45326424
Name: is_weekend, dtype: int64

=== is_holiday distribution ===
False    162169176
Name: is_holiday, dtype: int64

=== in_custom_event distribution ===
False    162169176
Name: in_custom_event, dtype: int64


process finished, exit codenote 0"""



import pandas as pd

# 1. load data(adjust the path as needed)
csv_path = '/finalproject/pems_data/step31_fillExter.csv'
df = pd.read_csv(csv_path, parse_dates=['timestamp'])

# 2. inspect the first rows
print("=== first 5 rowsdata ===")
print(df.head(), "\n")

# 3. data shape and field types
print("=== datanote ===")
print(df.shape, "\n")
print("=== field types ===")
print(df.dtypes, "\n")

# 4. missing valuesnote
print("=== missing valuesnote ===")
mis = df.isnull().sum().to_frame('missing_count')
mis['missing_pct'] = mis['missing_count'] / len(df) * 100
print(mis, "\n")

# 5. boolean field distribution
for col in ['is_weekend','is_holiday','in_custom_event']:
    print(f"=== {col} distribution ===")
    print(df[col].value_counts(dropna=False), "\n")

