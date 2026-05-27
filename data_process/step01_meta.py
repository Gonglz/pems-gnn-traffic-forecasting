#!/usr/bin/env python3
# coding: utf-8
"""
step01_meta.py

note:
  1. note D07 notedatafilenote
  2. note TopoMap Excel filenote
  3. notedatanote station_id note, generatenotedatanote:
  finalproject/
    ├─ data_process/         <-- notedirectory
    └─ pems_data/
        └─ pems_detector/    <-- datadirectory
            ├─ d07_text_meta_2023_12_22.txt
            └─ topomap.xlsx

outputfile:
  finalproject/pems_data/pems_detector/
    - step01_d07_meta.csv         note D07 notedata
    - step01_topomap_meta.csv     note TopoMap notedata
    - step01_station_meta.csv     notedatanote:
  station_id   note ID
  freeway      note
  direction    rowsnote(N/S/E/W)
  district     note
  county       note/note
  city         note
  state_pm     note(State PM)
  abs_pm       note(Absolute PM)
  latitude     note(note D07 datanote)
  longitude    note(note D07 datanote)
  length       note
  type         noteclassnote(note, note)
  lanes        note
  name         note
  sensor_type  noteclassnote(loops, radar note, note TopoMap)
  hov          note HOV note, note TopoMap

note:
  cd finalproject/data_process
  python step01_meta.py

notepath:
  python step01_meta.py \
    --d07../pems_data/pems_detector/d07_text_meta_2023_12_22.txt \
    --topo../pems_data/pems_detector/topomap.xlsx \
    --out_raw../pems_data/pems_detector/step01_d07_meta.csv \
    --out_topo../pems_data/pems_detector/step01_topomap_meta.csv \
    --out_merged../pems_data/pems_detector/step01_station_meta.csv
"""
import os
import pandas as pd
import argparse

# datadirectory: notedirectory pems_data/pems_detector
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', 'pems_data', 'pems_detector'
    )
)


def load_d07(path):
    """note D07 notedatafile"""
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
    """note TopoMap notedatafile"""
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
    parser = argparse.ArgumentParser(description='note, notedata')
    parser.add_argument('--d07', default=os.path.join(BASE_DIR, 'd07_text_meta_2023_12_22.txt'),
                        help='D07 notedatapath')
    parser.add_argument('--topo', default=os.path.join(BASE_DIR, 'topomap.xlsx'),
                        help='TopoMap notedatapath')
    parser.add_argument('--out_raw', default=os.path.join(BASE_DIR, 'step01_d07_meta.csv'),
                        help='output D07 note CSV path')
    parser.add_argument('--out_topo', default=os.path.join(BASE_DIR, 'step01_topomap_meta.csv'),
                        help='output TopoMap note CSV path')
    parser.add_argument('--out_merged', default=os.path.join(BASE_DIR, 'step01_station_meta.csv'),
                        help='outputnote CSV path')
    args = parser.parse_args()

    # noteinputfile
    for p in [args.d07, args.topo]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"notefile: {p}")

    # 1. note D07
    print(f'note D07 notedata: {args.d07}')
    d07_meta = load_d07(args.d07)
    d07_meta.to_csv(args.out_raw, index=False)
    print(f'✔ noteoutput D07 notedata: {args.out_raw}')

    # 2. note TopoMap
    print(f'note TopoMap notedata: {args.topo}')
    topo_meta = load_topomap(args.topo)
    topo_meta.to_csv(args.out_topo, index=False)
    print(f'✔ noteoutput TopoMap notedata: {args.out_topo}')

    # 3. notedata
    print('notedata(note station_id)')
    merged = pd.merge(
        d07_meta, topo_meta,
        on='station_id', how='outer',
        suffixes=('_d07', '_topo')
    )
    merged.to_csv(args.out_merged, index=False)
    print(f'✔ notegeneratenotedata: {args.out_merged}')

if __name__ == '__main__':
    main()