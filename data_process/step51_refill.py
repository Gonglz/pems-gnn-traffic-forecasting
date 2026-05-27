#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 51: safe refill for masked rows still missing after step 50."""

import argparse
import os
import sys

import pandas as pd


DEFAULT_RAW = '/scratch/lgong1/finalproject/pems_data/step31_fillExter.csv'
DEFAULT_MASK = '/scratch/lgong1/finalproject/pems_data/step34_maskMix.csv'
DEFAULT_INTERP = '/scratch/lgong1/finalproject/pems_data/step50_interpolated_fastest.parquet'
DEFAULT_OUTPUT = '/scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet'

FEATURES = ['flow', 'occupancy', 'speed']
INTERP_COLS = [f'{feat}_interp' for feat in FEATURES]
MASK_COLUMNS = ['mask_logic', 'mask_md', 'mask_hf']
BASE_KEYS = ['timestamp', 'station_id']


def _env(name):
    return os.environ.get(name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Refill masked rows that remain missing after step50 without using raw anomalous values.'
    )
    parser.add_argument('--raw', default=_env('STEP51_RAW') or DEFAULT_RAW)
    parser.add_argument('--mask', default=_env('STEP51_MASK') or DEFAULT_MASK)
    parser.add_argument('--interp', default=_env('STEP51_INTERP') or DEFAULT_INTERP)
    parser.add_argument('--output', default=_env('STEP51_OUTPUT') or DEFAULT_OUTPUT)
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
        missing = [col for col in INTERP_COLS if col not in df.columns]
        raw_missing = [col for col in FEATURES if col not in df.columns]
        raise ValueError(
            'Interpolation input must contain either step50 columns '
            f'{FEATURES} or step51 columns {INTERP_COLS}; missing {missing} / {raw_missing}'
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


def coerce_mask_bool(series):
    return series.astype('boolean').fillna(False).astype(bool)


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


def build_merged_frame(raw, mask, interp):
    raw_df = raw.copy()
    if 'timestamp' in raw_df.columns:
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    base_cols = BASE_KEYS + (['direction'] if 'direction' in raw_df.columns else [])
    base = raw_df[base_cols].drop_duplicates().reset_index(drop=True)

    mask_keys = merge_keys(base, mask)
    mask_small = aggregate_mask(mask, mask_keys)
    interp_small = aggregate_interp(interp, merge_keys(base, normalize_interp_columns(interp)))

    df = base.merge(mask_small, on=mask_keys, how='left', validate='m:1')
    df = df.merge(interp_small, on=merge_keys(base, interp_small), how='left', validate='m:1')
    df['mask'] = build_mask_flag(df)
    return df


def refill_missing_masked(df):
    out = df.copy()
    target_before = out['mask'] & out[INTERP_COLS].isna().any(axis=1)
    before = int(target_before.sum())

    timestamp_means = out.groupby('timestamp', dropna=False)[INTERP_COLS].transform('mean')
    for col in INTERP_COLS:
        fill_idx = out['mask'] & out[col].isna()
        out.loc[fill_idx, col] = timestamp_means.loc[fill_idx, col]

    group_cols = ['station_id'] + (['direction'] if 'direction' in out.columns else [])
    out['_row_id'] = range(len(out))
    ordered = out.sort_values(group_cols + ['timestamp', '_row_id'])
    temporal_values = ordered.groupby(group_cols, dropna=False)[INTERP_COLS].transform(
        lambda series: series.ffill().bfill()
    )
    temporal_values['_row_id'] = ordered['_row_id'].values
    temporal_values = temporal_values.set_index('_row_id')
    for col in INTERP_COLS:
        fill_idx = out['mask'] & out[col].isna()
        out.loc[fill_idx, col] = out.loc[fill_idx, '_row_id'].map(temporal_values[col])
    out = out.drop(columns=['_row_id'])

    target_after = out['mask'] & out[INTERP_COLS].isna().any(axis=1)
    after = int(target_after.sum())
    return out, before, after


def finalize_output(df, expected_unique_count):
    final = (
        df[BASE_KEYS + INTERP_COLS]
        .groupby(BASE_KEYS, as_index=False, dropna=False)[INTERP_COLS]
        .mean()
        .sort_values(BASE_KEYS)
        .reset_index(drop=True)
    )
    if len(final) != expected_unique_count:
        raise ValueError(
            f'Output row count {len(final)} does not match expected unique '
            f'(timestamp, station_id) count {expected_unique_count}'
        )
    return final[BASE_KEYS + INTERP_COLS]


def build_refilled_frame(raw, mask, interp):
    expected_unique_count = raw[BASE_KEYS].drop_duplicates().shape[0]
    merged = build_merged_frame(raw, mask, interp)
    refilled, _, _ = refill_missing_masked(merged)
    return finalize_output(refilled, expected_unique_count)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.raw, args.mask, args.interp):
        if not os.path.exists(path):
            print(f'Missing {path}', file=sys.stderr)
            sys.exit(1)

    raw = read_table(args.raw)
    mask = read_table(args.mask)
    interp = read_table(args.interp)

    expected_unique_count = raw[BASE_KEYS].drop_duplicates().shape[0]
    merged = build_merged_frame(raw, mask, interp)
    refilled, before, after = refill_missing_masked(merged)
    final = finalize_output(refilled, expected_unique_count)

    print(f'Masked rows still missing before refill: {before:,}')
    print(f'Masked rows still missing after refill: {after:,}')
    print(f'Output unique (timestamp, station_id) rows: {len(final):,}')

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    final.to_parquet(args.output, index=False)
    print(f'Wrote final interpolation result: {args.output}')


if __name__ == '__main__':
    main()
