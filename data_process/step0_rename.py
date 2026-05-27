#!/usr/bin/env python3
# coding: utf-8

import csv
from datetime import datetime
from pathlib import Path
import argparse
import sys

def parse_timestamp(ts_str: str) -> datetime:
    """note ISO, note MM/DD/YYYY HH:MM:SS"""
    for fmt in (None, "%m/%d/%Y %H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(ts_str)
            else:
                return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    raise ValueError(f"noteparsenote: {ts_str}")

def rename_files(data_dir: Path, pattern: str = "*.txt"):
    if not data_dir.exists():
        print(f"FAIL directorynote: {data_dir}")
        sys.exit(1)

    for file in sorted(data_dir.glob(pattern)):
        # note macOS notefilenotefile
        if file.name.startswith('.') or file.name.upper().startswith('__MACOSX'):
            continue
        try:
            # readnoterows(note)
            with file.open("r", newline="") as f:
                reader = csv.reader(f)
                row = next(reader)
            ts_str = row[0].strip()
            dt     = parse_timestamp(ts_str)
            date_str = dt.strftime("%Y%m%d")
            new_name = f"{date_str}_raw.txt"
            dst = file.with_name(new_name)
            if dst.exists():
                print(f"⚠️ note, note: {dst.name}")
                continue
            file.rename(dst)
            print(f"PASS {file.name} -> {dst.name}")
        except Exception as e:
            print(f"FAIL notefailed {file.name}: {e}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_dir  = project_root / "pems_data" / "pems_dataset"

    parser = argparse.ArgumentParser(
        description="notefilenote.txt note YYYYMMDD_raw.txt"
    )
    parser.add_argument(
        "--dir", "-d",
        default=str(default_dir),
        help=f"datanotedirectory(default: {default_dir})"
    )
    args = parser.parse_args()
    data_dir = Path(args.dir)
    print(f"➡️ notedirectory: {data_dir}")
    rename_files(data_dir)
