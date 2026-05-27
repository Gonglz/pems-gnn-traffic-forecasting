import pandas as pd
import matplotlib.pyplot as plt

# 1. readnote
df_health = pd.read_csv('/finalproject/pems_data/step20_day_health_factor_GPU.csv', parse_dates=['date'])

# 2. outputnote
print(df_health['health_conf'].describe())

# 3. note
plt.figure(figsize=(8, 4))
plt.hist(df_health['health_conf'], bins=50)
plt.xlabel('Health Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Health Weights')
plt.tight_layout()
plt.show()


""""
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step#healthVerify.py
count    567008.000000
mean          0.520622
std           0.453110
min           0.000000
25%           0.007748
50%           0.666666
75%           0.996108
max           1.000000
"""