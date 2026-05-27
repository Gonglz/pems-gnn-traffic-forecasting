#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step21_fillHealth_optimized.py

note:
 - readnote step1_raw_long.csv, note chunk note
 - note day_health_factor.csv
 - note health_conf note station, note 0.0039
 - outputnoteresultnote step2_filled.csv
 - note

"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_LONG = BASE_DIR / "pems_data" / "step1_raw_long.csv"
DAY_HEALTH = BASE_DIR / "pems_data" / "step20_day_health_factor_GPU.csv"
OUT_FILE = BASE_DIR / "pems_data" / "step21_fillHealth.csv"

# note chunk noterowsnote, note
CHUNKSIZE = 5_000_000
DEFAULT_HEALTH = 0.0039

# ─── notefunction ─────────────────────────────────────────────────────────────
def main():
    # readnote
    dfh = pd.read_csv(DAY_HEALTH, parse_dates=['date'])

    # computenoterowsnote, note
    total_rows = sum(1 for _ in open(RAW_LONG, 'r', encoding='utf-8')) - 1
    reader = pd.read_csv(RAW_LONG, parse_dates=['timestamp'], chunksize=CHUNKSIZE)

    first_chunk = True
    with tqdm(total=total_rows, unit='rows', desc='Processing') as pbar:
        for chunk in reader:
            # generatenote
            chunk['date'] = chunk['timestamp'].dt.floor('D')
            # note
            chunk = chunk.merge(dfh, on=['station_id','date'], how='left')
            # notedefault
            chunk['health_conf'] = chunk['health_conf'].fillna(DEFAULT_HEALTH)
            # noteoutput
            chunk.to_csv(
                OUT_FILE,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False,
                encoding='utf-8'
            )
            first_chunk = False
            pbar.update(len(chunk))

if __name__ == '__main__':
    main()
