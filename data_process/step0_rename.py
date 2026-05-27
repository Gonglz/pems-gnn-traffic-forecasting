#!/usr/bin/env python3
# coding: utf-8

import csv
from datetime import datetime
from pathlib import Path
import argparse
import sys

def parse_timestamp(ts_str: str) -> datetime:
    """先尝试 ISO，再尝试 MM/DD/YYYY HH:MM:SS"""
    for fmt in (None, "%m/%d/%Y %H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(ts_str)
            else:
                return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    raise ValueError(f"无法解析时间戳: {ts_str}")

def rename_files(data_dir: Path, pattern: str = "*.txt"):
    if not data_dir.exists():
        print(f"❌ 目录不存在：{data_dir}")
        sys.exit(1)

    for file in sorted(data_dir.glob(pattern)):
        # 跳过 macOS 系统文件夹或隐藏文件
        if file.name.startswith('.') or file.name.upper().startswith('__MACOSX'):
            continue
        try:
            # 读取第一行（若有表头请调整跳过）
            with file.open("r", newline="") as f:
                reader = csv.reader(f)
                row = next(reader)
            ts_str = row[0].strip()
            dt     = parse_timestamp(ts_str)
            date_str = dt.strftime("%Y%m%d")
            new_name = f"{date_str}_raw.txt"
            dst = file.with_name(new_name)
            if dst.exists():
                print(f"⚠️ 已存在，跳过: {dst.name}")
                continue
            file.rename(dst)
            print(f"✅ {file.name} → {dst.name}")
        except Exception as e:
            print(f"❌ 处理失败 {file.name}: {e}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_dir  = project_root / "pems_data" / "pems_dataset"

    parser = argparse.ArgumentParser(
        description="按文件内首条时间戳重命名所有 .txt 为 YYYYMMDD_raw.txt"
    )
    parser.add_argument(
        "--dir", "-d",
        default=str(default_dir),
        help=f"数据所在目录（默认: {default_dir}）"
    )
    args = parser.parse_args()
    data_dir = Path(args.dir)
    print(f"➡️ 正在处理目录: {data_dir}")
    rename_files(data_dir)
