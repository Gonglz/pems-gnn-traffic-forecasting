#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# ─── 1. directoryconfiguration ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent              #.../finalproject/data_process
PROJECT_DIR   = SCRIPT_DIR.parent                             #.../finalproject
PEMS_ROOT     = PROJECT_DIR / "pems_data"                     #.../finalproject/pems_data
RAW_DIR       = PEMS_ROOT / "pems_dataset"                    # note raw.txt note
OUT_LONG      = PEMS_ROOT / "step1_raw_long.csv"              # outputnote
OUT_META      = PEMS_ROOT / "step1_station_meta.csv"          # outputnotedata

# ─── 2. notefilenote ────────────────────────────────────────────────────────────────
def process_file(raw_file: Path) -> pd.DataFrame:
    usecols = [0,1,3,4,5,6,7,8,9,10,11]
    df = pd.read_csv(raw_file, header=None, usecols=usecols)
    df.columns = [
        "timestamp",
        "station_id",
        "freeway_id",
        "direction",
        "lane_type",
        "station_length",
        "samples",
        "pct_observed",
        "flow",
        "occupancy",
        "speed",
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S")
    return df

# ─── 3. noteworkflow ────────────────────────────────────────────────────────────────
def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"notedatadirectory: {RAW_DIR}")

    # 3.1 noteread
    frames = []
    for raw in sorted(RAW_DIR.glob("*_raw.txt")):
        print(f"Reading {raw.name} …")
        frames.append(process_file(raw))
    data = pd.concat(frames, ignore_index=True)
    print(f"-> noteread {len(data)} note")

    # 3.2 outputnote
    long_df = data[[
        "timestamp","station_id","freeway_id","direction",
        "lane_type","station_length","samples",
        "pct_observed","flow","occupancy","speed",
    ]]
    long_df.to_csv(OUT_LONG, index=False)
    print(f"-> note: {OUT_LONG}")

    # 3.3 outputstaticnotedata
    meta_df = data[["station_id","lane_type","station_length"]].drop_duplicates()
    meta_df.to_csv(OUT_META, index=False)
    print(f"-> notedata: {OUT_META}")

if __name__ == "__main__":
    main()
