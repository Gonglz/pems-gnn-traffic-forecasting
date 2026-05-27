#!/usr/bin/env python3
# coding: utf-8
"""
step01_meta.py

功能：
  1. 从 D07 探测器元数据文件中提取关键字段并清洗
  2. 从 TopoMap Excel 文件中提取关键字段并清洗
  3. 将两份清洗后的元数据按 station_id 合并，生成一份综合的探测器元数据表

脚本假设项目结构：
  finalproject/
    ├─ data_process/         <-- 本脚本所在目录
    └─ pems_data/
        └─ pems_detector/    <-- 数据目录
            ├─ d07_text_meta_2023_12_22.txt
            └─ topomap.xlsx

输出文件：
  finalproject/pems_data/pems_detector/
    - step01_d07_meta.csv         清洗后的 D07 元数据
    - step01_topomap_meta.csv     清洗后的 TopoMap 元数据
    - step01_station_meta.csv     两者合并后的综合元数据表

各列含义：
  station_id   探测器唯一 ID
  freeway      高速编号或名称
  direction    行驶方向（N/S/E/W）
  district     区域编号
  county       县/区
  city         城市
  state_pm     州内里程标（State PM）
  abs_pm       绝对里程标（Absolute PM）
  latitude     纬度（仅 D07 数据来源）
  longitude    经度（仅 D07 数据来源）
  length       探测器覆盖路段长度
  type         探测器类型（如主线、匝道等）
  lanes        探测车道数
  name         探测器名称或位置描述
  sensor_type  传感器类型（loops、radar 等，仅 TopoMap）
  hov          是否 HOV 专用，仅 TopoMap

用法示例:
  cd finalproject/data_process
  python step01_meta.py

可手动指定路径：
  python step01_meta.py \
    --d07 ../pems_data/pems_detector/d07_text_meta_2023_12_22.txt \
    --topo ../pems_data/pems_detector/topomap.xlsx \
    --out_raw ../pems_data/pems_detector/step01_d07_meta.csv \
    --out_topo ../pems_data/pems_detector/step01_topomap_meta.csv \
    --out_merged ../pems_data/pems_detector/step01_station_meta.csv
"""
import os
import pandas as pd
import argparse

# 数据目录：相对于本脚本的上两级目录 pems_data/pems_detector
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', 'pems_data', 'pems_detector'
    )
)


def load_d07(path):
    """加载并清洗 D07 元数据文件"""
    df = pd.read_csv(path, sep='[\t,]', engine='python', dtype=str)
    columns_map = {
        'ID': 'station_id',
        'Fwy': 'freeway',
        'Dir': 'direction',
        'District': 'district',
        'County': 'county',
        'City': 'city',
        'State_PM': 'state_pm',
        'Abs_PM': 'abs_pm',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Length': 'length',
        'Type': 'type',
        'Lanes': 'lanes',
        'Name': 'name'
    }
    df = df.rename(columns=columns_map)
    return df[list(columns_map.values())].copy()


def load_topomap(path):
    """加载并清洗 TopoMap 元数据文件"""
    df = pd.read_excel(path, dtype=str)
    columns_map = {
        'ID': 'station_id',
        'Fwy': 'freeway',
        'District': 'district',
        'County': 'county',
        'City': 'city',
        'CA PM': 'state_pm',
        'Abs PM': 'abs_pm',
        'Length': 'length',
        'Name': 'name',
        'Lanes': 'lanes',
        'Type': 'type',
        'Sensor Type': 'sensor_type',
        'HOV': 'hov'
    }
    df = df.rename(columns=columns_map)
    return df[list(columns_map.values())].copy()


def main():
    parser = argparse.ArgumentParser(description='提取、清洗并合并探测器元数据')
    parser.add_argument('--d07', default=os.path.join(BASE_DIR, 'd07_text_meta_2023_12_22.txt'),
                        help='D07 源数据路径')
    parser.add_argument('--topo', default=os.path.join(BASE_DIR, 'topomap.xlsx'),
                        help='TopoMap 源数据路径')
    parser.add_argument('--out_raw', default=os.path.join(BASE_DIR, 'step01_d07_meta.csv'),
                        help='输出 D07 清洗后 CSV 路径')
    parser.add_argument('--out_topo', default=os.path.join(BASE_DIR, 'step01_topomap_meta.csv'),
                        help='输出 TopoMap 清洗后 CSV 路径')
    parser.add_argument('--out_merged', default=os.path.join(BASE_DIR, 'step01_station_meta.csv'),
                        help='输出合并后 CSV 路径')
    args = parser.parse_args()

    # 校验输入文件
    for p in [args.d07, args.topo]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    # 1. 处理 D07
    print(f'开始处理 D07 元数据: {args.d07}')
    d07_meta = load_d07(args.d07)
    d07_meta.to_csv(args.out_raw, index=False)
    print(f'✔ 已输出 D07 清洗数据: {args.out_raw}')

    # 2. 处理 TopoMap
    print(f'开始处理 TopoMap 元数据: {args.topo}')
    topo_meta = load_topomap(args.topo)
    topo_meta.to_csv(args.out_topo, index=False)
    print(f'✔ 已输出 TopoMap 清洗数据: {args.out_topo}')

    # 3. 合并两份元数据
    print('开始合并两份元数据（按 station_id）')
    merged = pd.merge(
        d07_meta, topo_meta,
        on='station_id', how='outer',
        suffixes=('_d07', '_topo')
    )
    merged.to_csv(args.out_merged, index=False)
    print(f'✔ 已生成合并元数据: {args.out_merged}')

if __name__ == '__main__':
    main()