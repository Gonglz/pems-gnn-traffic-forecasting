import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. notedata ---
base = '/scratch/lgong1/finalproject/pems_data'
# notefilenote day_health_factor.csv note step20_day_health_factor_GPU.csv
fname = 'step20_day_health_factor_GPU.csv'
path = os.path.join(base, fname)
df = pd.read_csv(path, parse_dates=['date'])

# --- 2. note ---
print("=== Summary Statistics ===")
print(df['health_conf'].describe())

# --- 3. note ---
plt.figure(figsize=(6,4))
plt.hist(df['health_conf'], bins=50, edgecolor='k')
plt.xlabel('health_conf')
plt.ylabel('Count')
plt.title('Distribution of Day-level Health Factor')
plt.tight_layout()
plt.show()

# --- 4. notedistributionfunction(CDF) ---
vals = np.sort(df['health_conf'].values)
cdf = np.linspace(0, 1, len(vals))
plt.figure(figsize=(6,4))
plt.plot(cdf, vals)
plt.xlabel('Fraction of Samples')
plt.ylabel('health_conf')
plt.title('CDF of Day-level Health Factor')
plt.tight_layout()
plt.show()

# --- 5. note ---
for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    q = np.quantile(vals, p)
    print(f"{int(p*100):>2d}th percentile: {q:.4f}")

"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step20_healthCheck.py
=== Summary Statistics ===
count    567008.000000
mean          0.520622
std           0.453110
min           0.000000
25%           0.007748
50%           0.666666
75%           0.996108
max           1.000000
Name: health_conf, dtype: float64
 1th percentile: 0.0000
 5th percentile: 0.0000
10th percentile: 0.0000
25th percentile: 0.0077
50th percentile: 0.6667
75th percentile: 0.9961
90th percentile: 1.0000
95th percentile: 1.0000
99th percentile: 1.0000

process finished, exit codenote 0
"""