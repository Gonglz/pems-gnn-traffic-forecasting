#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step34_maskMix.py

note:
 1. readnote:
    - note(step30_logic_mask_continuous.csv)
    - Mahalanobis note(step32_md.csv)
    - note(day_health_factor.csv)
 2. note, generatenote mask_mix:
    timestamp, station_id, mask_logic, mask_md, mask_hf
 3. output CSV note Parquet, note:
    cd finalproject/data_process
    python step34_maskMix.py

note:
    pip install pandas pyarrow fastparquet

    /scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/data_process/step34_maskMix.py
1. readnote…
2. read Mahalanobis note…
3. note & MD note…
4. readnotegenerate mask_hf…
5. note, note mask_hf…
6. note, output CSV & Parquet…
✔ notesavenotefile:
  CSV -> /scratch/lgong1/finalproject/pems_data/step34_maskMix.csv
  Parquet -> /scratch/lgong1/finalproject/pems_data/step34_maskMix.parquet

"""
import os
import pandas as pd

# --- note & path ---
HF_THRESH    = 0.3  # note
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pems_data'))
LOGIC_CSV    = os.path.join(BASE_DIR, 'step30_logic_mask_continuous.csv')
MD_CSV       = os.path.join(BASE_DIR, 'step32_md.csv')
HF_CSV       = os.path.join(BASE_DIR, 'step20_day_health_factor_GPU.csv')
OUT_CSV      = os.path.join(BASE_DIR, 'step34_maskMix.csv')
OUT_PARQUET  = os.path.join(BASE_DIR, 'step34_maskMix.parquet')

def main():
    print("1. readnote…")
    logic = pd.read_csv(
        LOGIC_CSV,
        usecols=['timestamp','station_id','mask_logic'],
        parse_dates=['timestamp']
    )

    print("2. read Mahalanobis note…")
    md = pd.read_csv(
        MD_CSV,
        usecols=['timestamp','station_id','mask_md'],
        parse_dates=['timestamp']
    )

    print("3. note & MD note…")
    df = logic.merge(
        md,
        on=['timestamp','station_id'],
        how='left'
    )
    # missing valuesnote False
    df['mask_md'] = df['mask_md'].fillna(False)

    print("4. readnotegenerate mask_hf…")
    hf = pd.read_csv(
        HF_CSV,
        parse_dates=['date']
    )
    # note 'health_factor', note
    if 'health_factor' not in hf.columns:
        hf = hf.rename(columns={hf.columns[-1]: 'health_factor'})
    hf['mask_hf'] = hf['health_factor'] < HF_THRESH
    hf = hf[['station_id','date','mask_hf']]

    print("5. note, note mask_hf…")
    # note
    df['date'] = df['timestamp'].dt.date
    # note
    df = df.merge(
        hf.assign(date=hf['date'].dt.date),
        on=['station_id','date'],
        how='left'
    )
    df['mask_hf'] = df['mask_hf'].fillna(False)
    # note
    df.drop(columns=['date'], inplace=True)

    print("6. note, output CSV & Parquet…")
    df = df[['timestamp','station_id','mask_logic','mask_md','mask_hf']]
    df.to_csv(OUT_CSV, index=False)
    # Parquet filenote, noteread
    df.to_parquet(OUT_PARQUET, index=False, compression='snappy')

    print(f"✔ notesavenotefile: \n  CSV -> {OUT_CSV}\n  Parquet -> {OUT_PARQUET}")

if __name__ == '__main__':
    main()
