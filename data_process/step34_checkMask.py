import pandas as pd
from itertools import combinations

# 1. 读取 CSV
df = pd.read_csv('/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv', parse_dates=['timestamp'])
total = len(df)

# 2. 三种掩码列名
mask_cols = ['mask_logic', 'mask_md', 'mask_hf']

print(f"总记录数：{total}\n")

# 3. 单独掩码统计
print("== 单独掩码 ==")
for col in mask_cols:
    cnt = df[col].sum()
    pct = cnt / total * 100
    print(f"{col:12s}：{cnt:6d} 条，{pct:5.2f}%")

# 4. 两两交集
print("\n== 两两交集 ==")
for a, b in combinations(mask_cols, 2):
    both = df[a] & df[b]
    cnt  = both.sum()
    pct  = cnt / total * 100
    print(f"{a} & {b:8s}：{cnt:6d} 条，{pct:5.2f}%")

# 5. 三者全交集
print("\n== 全交集 ==")
all3 = df[mask_cols].all(axis=1).sum()
pct3 = all3 / total * 100
print(f"mask_logic & mask_md & mask_hf：{all3:6d} 条，{pct3:5.2f}%")


"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_checkMask.py (0.5)
总记录数：162169176

== 单独掩码 ==
mask_logic  ：29105592 条，17.95%
mask_md     ：1207499 条， 0.74%
mask_hf     ：76770732 条，47.34%

== 两两交集 ==
mask_logic & mask_md ：629683 条， 0.39%
mask_logic & mask_hf ：11790162 条， 7.27%
mask_md & mask_hf ：631173 条， 0.39%

== 全交集 ==
mask_logic & mask_md & mask_hf：248454 条， 0.15%


/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_checkMask.py (0.3)
总记录数：162169176

== 单独掩码 ==
mask_logic  ：29105592 条，17.95%
mask_md     ：2577031 条， 1.59%
mask_hf     ：75249804 条，46.40%

== 两两交集 ==
mask_logic & mask_md ：956578 条， 0.59%
mask_logic & mask_hf ：11485562 条， 7.08%
mask_md & mask_hf ：1305646 条， 0.81%

== 全交集 ==
mask_logic & mask_md & mask_hf：376856 条， 0.23%

进程已结束，退出代码为 0

进程已结束，退出代码为 0"""