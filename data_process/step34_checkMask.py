import pandas as pd
from itertools import combinations

# 1. read CSV
df = pd.read_csv('/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv', parse_dates=['timestamp'])
total = len(df)

# 2. note
mask_cols = ['mask_logic', 'mask_md', 'mask_hf']

print(f"note: {total}\n")

# 3. note
print("== note ==")
for col in mask_cols:
    cnt = df[col].sum()
    pct = cnt / total * 100
    print(f"{col:12s}: {cnt:6d} note, {pct:5.2f}%")

# 4. note
print("\n== note ==")
for a, b in combinations(mask_cols, 2):
    both = df[a] & df[b]
    cnt  = both.sum()
    pct  = cnt / total * 100
    print(f"{a} & {b:8s}: {cnt:6d} note, {pct:5.2f}%")

# 5. note
print("\n== note ==")
all3 = df[mask_cols].all(axis=1).sum()
pct3 = all3 / total * 100
print(f"mask_logic & mask_md & mask_hf: {all3:6d} note, {pct3:5.2f}%")


"""/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_checkMask.py (0.5)
note: 162169176

== note ==
mask_logic: 29105592 note, 17.95%
mask_md: 1207499 note,  0.74%
mask_hf: 76770732 note, 47.34%

== note ==
mask_logic & mask_md: 629683 note,  0.39%
mask_logic & mask_hf: 11790162 note,  7.27%
mask_md & mask_hf: 631173 note,  0.39%

== note ==
mask_logic & mask_md & mask_hf: 248454 note,  0.15%


/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_checkMask.py (0.3)
note: 162169176

== note ==
mask_logic: 29105592 note, 17.95%
mask_md: 2577031 note,  1.59%
mask_hf: 75249804 note, 46.40%

== note ==
mask_logic & mask_md: 956578 note,  0.59%
mask_logic & mask_hf: 11485562 note,  7.08%
mask_md & mask_hf: 1305646 note,  0.81%

== note ==
mask_logic & mask_md & mask_hf: 376856 note,  0.23%

process finished, exit codenote 0

process finished, exit codenote 0"""