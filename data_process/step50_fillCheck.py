#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check masked-fill completeness and unmasked consistency for step50/step51 outputs."""

import argparse
import os
import sys

import numpy as np
import pandas as pd


DEFAULT_RAW = '/scratch/lgong1/finalproject/pems_data/step31_fillExter.csv'
DEFAULT_MASK = '/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv'
DEFAULT_INTERP = '/scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet'

FEATURES = ['flow', 'occupancy', 'speed']
INTERP_COLS = [f'{feat}_interp' for feat in FEATURES]
MASK_COLUMNS = ['mask_logic', 'mask_md', 'mask_hf']
BASE_KEYS = ['timestamp', 'station_id']


def _env(name):
    return os.environ.get(name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Print row, duplicate, masked-missing, and unmasked-diff checks.'
    )
    parser.add_argument('--raw', default=_env('FILL_CHECK_RAW') or DEFAULT_RAW)
    parser.add_argument('--mask', default=_env('FILL_CHECK_MASK') or DEFAULT_MASK)
    parser.add_argument('--interp', default=_env('FILL_CHECK_INTERP') or DEFAULT_INTERP)
    return parser.parse_args(argv)


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.parquet', '.pq'}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def normalize_interp_columns(interp):
    df = interp.copy()
    if all(col in df.columns for col in INTERP_COLS):
        pass
    elif all(col in df.columns for col in FEATURES):
        df = df.rename(columns={feat: f'{feat}_interp' for feat in FEATURES})
    else:
        raise ValueError(
            f'Interpolation data must contain either {FEATURES} or {INTERP_COLS}'
        )
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    keep = BASE_KEYS + (['direction'] if 'direction' in df.columns else []) + INTERP_COLS
    return df[keep]


def merge_keys(left, right):
    keys = list(BASE_KEYS)
    if 'direction' in left.columns and 'direction' in right.columns:
        keys.append('direction')
    return keys


def duplicate_count(df, keys):
    if not all(col in df.columns for col in keys):
        return None
    return int(df.duplicated(keys).sum())


def coerce_mask_bool(series):
    return series.astype('boolean').fillna(False).astype(bool)


def aggregate_raw(raw):
    df = raw.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    keys = BASE_KEYS + (['direction'] if 'direction' in df.columns else [])
    raw_features = [feat for feat in FEATURES if feat in df.columns]
    if raw_features:
        return (
            df[keys + raw_features]
            .groupby(keys, as_index=False, dropna=False)[raw_features]
            .mean()
            .reset_index(drop=True)
        )
    return df[keys].drop_duplicates().reset_index(drop=True)


def aggregate_mask(mask, keys):
    cols = [col for col in keys + MASK_COLUMNS if col in mask.columns]
    df = mask[cols].copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    present_masks = [col for col in MASK_COLUMNS if col in df.columns]
    if not present_masks:
        return df[keys].drop_duplicates().reset_index(drop=True)
    for col in present_masks:
        df[col] = coerce_mask_bool(df[col])
    return (
        df.groupby(keys, as_index=False, dropna=False)[present_masks]
        .max()
        .reset_index(drop=True)
    )


def aggregate_interp(interp, keys):
    df = normalize_interp_columns(interp)
    return (
        df.groupby(keys, as_index=False, dropna=False)[INTERP_COLS]
        .mean()
        .reset_index(drop=True)
    )


def build_mask_flag(df):
    present_masks = [col for col in MASK_COLUMNS if col in df.columns]
    if not present_masks:
        return pd.Series(False, index=df.index)
    mask = coerce_mask_bool(df[present_masks[0]])
    for col in present_masks[1:]:
        mask = mask | coerce_mask_bool(df[col])
    return mask


def compute_check_stats(raw, mask, interp):
    raw_base = aggregate_raw(raw)
    interp_norm = normalize_interp_columns(interp)
    mask_keys = merge_keys(raw_base, mask)
    interp_keys = merge_keys(raw_base, interp_norm)

    mask_small = aggregate_mask(mask, mask_keys)
    interp_small = aggregate_interp(interp_norm, interp_keys)

    df = raw_base.merge(mask_small, on=mask_keys, how='left', validate='m:1')
    df = df.merge(interp_small, on=interp_keys, how='left', validate='m:1')
    df['mask'] = build_mask_flag(df)

    raw_feature_cols = [feat for feat in FEATURES if feat in df.columns]
    unmasked_max_diffs = {}
    for feat in raw_feature_cols:
        interp_col = f'{feat}_interp'
        diff = (df.loc[~df['mask'], feat] - df.loc[~df['mask'], interp_col]).abs().max()
        unmasked_max_diffs[feat] = float(diff) if pd.notna(diff) else np.nan

    missing_masked_count = int((df['mask'] & df[INTERP_COLS].isna().any(axis=1)).sum())
    mask_count = int(df['mask'].sum())
    expected_unique = int(raw[BASE_KEYS].drop_duplicates().shape[0])

    return {
        'row_counts': {
            'raw': int(len(raw)),
            'mask': int(len(mask)),
            'interp': int(len(interp)),
            'merged': int(len(df)),
            'expected_unique_timestamp_station': expected_unique,
        },
        'duplicate_counts': {
            'raw_timestamp_station': duplicate_count(raw, BASE_KEYS),
            'mask_timestamp_station': duplicate_count(mask, BASE_KEYS),
            'interp_timestamp_station': duplicate_count(interp_norm, BASE_KEYS),
            'raw_merge_key': duplicate_count(raw, list(raw_base.columns.intersection(BASE_KEYS + ['direction']))),
            'mask_merge_key': duplicate_count(mask, mask_keys),
            'interp_merge_key': duplicate_count(interp_norm, interp_keys),
        },
        'mask_count': mask_count,
        'unmasked_count': int(len(df) - mask_count),
        'missing_masked_count': missing_masked_count,
        'unmasked_max_diffs': unmasked_max_diffs,
    }


def print_stats(stats):
    rows = stats['row_counts']
    print('Row counts:')
    print(f"  raw: {rows['raw']:,}")
    print(f"  mask: {rows['mask']:,}")
    print(f"  interp: {rows['interp']:,}")
    print(f"  merged check rows: {rows['merged']:,}")
    print(f"  expected unique (timestamp, station_id): {rows['expected_unique_timestamp_station']:,}")

    print('\nDuplicate counts:')
    for name, value in stats['duplicate_counts'].items():
        display = 'n/a' if value is None else f'{value:,}'
        print(f'  {name}: {display}')

    print('\nMasked completeness:')
    print(f"  masked rows: {stats['mask_count']:,}")
    print(f"  masked rows still missing any interpolated feature: {stats['missing_masked_count']:,}")
    print(f"  unmasked rows: {stats['unmasked_count']:,}")

    print('\nUnmasked max diffs:')
    if not stats['unmasked_max_diffs']:
        print('  skipped: raw flow/occupancy/speed columns are not available')
    else:
        for feat in FEATURES:
            if feat in stats['unmasked_max_diffs']:
                print(f"  {feat:11s}: {stats['unmasked_max_diffs'][feat]}")


def main(argv=None):
    args = parse_args(argv)
    for path in (args.raw, args.mask, args.interp):
        if not os.path.exists(path):
            print(f'Missing {path}', file=sys.stderr)
            sys.exit(1)

    raw = read_table(args.raw)
    mask = read_table(args.mask)
    interp = read_table(args.interp)
    print_stats(compute_check_stats(raw, mask, interp))


if __name__ == '__main__':
    main()
