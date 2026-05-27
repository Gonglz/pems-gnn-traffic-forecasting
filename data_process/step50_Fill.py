#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 50: GPU-assisted interpolation for masked traffic observations."""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import cudf
    import cupy as cp
    import dask_cudf
    from cuml.neighbors import NearestNeighbors
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    from numba import cuda
except ImportError as exc:  # Allows smoke tests to import pure helpers locally.
    cudf = None
    cp = None
    dask_cudf = None
    NearestNeighbors = None
    Client = None
    LocalCUDACluster = None
    cuda = None
    GPU_IMPORT_ERROR = exc
else:
    GPU_IMPORT_ERROR = None


DEFAULT_BASE_DIR = '/scratch/lgong1/finalproject/pems_data'
DEFAULT_RAW_PARQ = os.path.join(DEFAULT_BASE_DIR, 'step31_fillExter.parquet')
DEFAULT_MASK_PARQ = os.path.join(DEFAULT_BASE_DIR, 'step34_maskMix.parquet')
DEFAULT_STATIONS_CSV = os.path.join(DEFAULT_BASE_DIR, 'step01_d07_meta.csv')
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, 'step50_interpolated_fastest.parquet')

THREADS = 256
FEATURES = ['flow', 'occupancy', 'speed']
MASK_COLUMNS = ['mask_logic', 'mask_md', 'mask_hf']
BASE_OUTPUT_COLUMNS = ['timestamp', 'station_id', 'direction'] + FEATURES
K_NEIGHBORS = 8
DEFAULT_GPU_DEVICES = '0,1,2,3'
DEFAULT_NUM_WORKERS = 4
REPARTITIONS = 2


@dataclass
class Step50Config:
    base_dir: str
    raw_parq: str
    mask_parq: str
    stations_csv: str
    output_dir: str
    gpu_devices: str
    num_workers: int
    sample_mode: bool


def _env(name):
    return os.environ.get(name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fill masked traffic rows with local, global, temporal, and final fallback passes.'
    )
    parser.add_argument('--base-dir', default=_env('STEP50_BASE_DIR') or DEFAULT_BASE_DIR)
    parser.add_argument('--raw-parq', default=_env('STEP50_RAW_PARQ'))
    parser.add_argument('--mask-parq', default=_env('STEP50_MASK_PARQ'))
    parser.add_argument('--stations-csv', default=_env('STEP50_STATIONS_CSV'))
    parser.add_argument('--output-dir', default=_env('STEP50_OUTPUT_DIR'))
    parser.add_argument('--gpu-devices', default=_env('STEP50_GPU_DEVICES') or DEFAULT_GPU_DEVICES)
    parser.add_argument('--num-workers', type=int, default=int(_env('STEP50_NUM_WORKERS') or DEFAULT_NUM_WORKERS))
    parser.add_argument(
        '--sample-mode',
        action='store_true',
        default=(_env('STEP50_SAMPLE_MODE') or '').lower() in {'1', 'true', 'yes', 'y'},
        help='Use fewer repartitions for small local/sample runs.',
    )
    args = parser.parse_args(argv)

    base_dir = args.base_dir
    return Step50Config(
        base_dir=base_dir,
        raw_parq=args.raw_parq or os.path.join(base_dir, 'step31_fillExter.parquet'),
        mask_parq=args.mask_parq or os.path.join(base_dir, 'step34_maskMix.parquet'),
        stations_csv=args.stations_csv or os.path.join(base_dir, 'step01_d07_meta.csv'),
        output_dir=args.output_dir or os.path.join(base_dir, 'step50_interpolated_fastest.parquet'),
        gpu_devices=args.gpu_devices,
        num_workers=args.num_workers,
        sample_mode=args.sample_mode,
    )


def require_gpu_dependencies():
    if GPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            'step50_Fill.py requires cudf, cupy, dask_cudf, dask_cuda, cuml, and numba '
            'for full data jobs. The pure helper functions remain importable for smoke tests.'
        ) from GPU_IMPORT_ERROR


if cuda is not None:

    @cuda.jit
    def global_kernel(feat, mask_flag, sid, did, grp_idx, grp_val, G, out):
        i = cuda.grid(1)
        if i < feat.shape[0] and mask_flag[i]:
            s = sid[i]
            d = did[i]
            for g in range(G):
                if grp_idx[g, 0] == s and grp_idx[g, 1] == d:
                    out[i] = grp_val[g]
                    break

    @cuda.jit
    def temporal_kernel(ts, mask_flag, offs, out):
        seg = cuda.blockIdx.x
        start = offs[seg]
        end = offs[seg + 1]
        tid = cuda.threadIdx.x
        stride = cuda.blockDim.x
        for idx in range(start + tid, end, stride):
            if mask_flag[idx]:
                prev = idx - 1
                while prev >= start and mask_flag[prev]:
                    prev -= 1
                nxt = idx + 1
                while nxt < end and mask_flag[nxt]:
                    nxt += 1
                if prev >= start and nxt < end:
                    t0 = ts[prev]
                    t1 = ts[nxt]
                    v0 = out[prev]
                    v1 = out[nxt]
                    ratio = (ts[idx] - t0) / (t1 - t0)
                    out[idx] = v0 + (v1 - v0) * ratio
                elif prev >= start:
                    out[idx] = out[prev]
                elif nxt < end:
                    out[idx] = out[nxt]

else:
    global_kernel = None
    temporal_kernel = None


def parquet_columns(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return list(pq.ParquetFile(path).schema.names)


def build_mask_flag(df, features=FEATURES, mask_columns=MASK_COLUMNS):
    mask_cols = [col for col in mask_columns if col in df.columns]
    if mask_cols:
        mask_flag = df[mask_cols[0]].fillna(False).astype('bool')
        for col in mask_cols[1:]:
            mask_flag = mask_flag | df[col].fillna(False).astype('bool')
    else:
        mask_flag = df[features[0]].isna() & False

    for feat in features:
        mask_flag = mask_flag | df[feat].isna()
    return mask_flag


def initialize_masked_features(df, features=FEATURES, mask_columns=MASK_COLUMNS):
    """Add _mask_flag and clear masked feature values so anomalous raw values cannot leak."""
    out = df.copy()
    out['_mask_flag'] = build_mask_flag(out, features=features, mask_columns=mask_columns)
    for feat in features:
        out[feat] = out[feat].where(~out['_mask_flag'], np.nan)
    return out


def apply_partition_local_fill(pdf, nbr_map, features=FEATURES):
    """Fill masked rows from same-timestamp neighbor station values available in this partition."""
    if len(pdf) == 0:
        return pdf

    out = pdf.copy().reset_index(drop=True)
    if '_mask_flag' not in out.columns:
        out = initialize_masked_features(out, features=features)

    if len(nbr_map) == 0:
        return out

    out['_row_id'] = np.arange(len(out), dtype=np.int64)
    masked = out.loc[out['_mask_flag'], ['_row_id', 'timestamp', 'station_id']]
    if len(masked) == 0:
        return out.drop(columns=['_row_id'])

    pairs = nbr_map[['station_id', 'neighbor_station_id']]
    candidates = masked.merge(pairs, on='station_id', how='inner')
    if len(candidates) == 0:
        return out.drop(columns=['_row_id'])

    neighbor_cols = {feat: f'_nbr_{feat}' for feat in features}
    valid_values = (
        out.loc[~out['_mask_flag'], ['timestamp', 'station_id'] + list(features)]
        .rename(columns={'station_id': 'neighbor_station_id', **neighbor_cols})
    )
    candidates = candidates.merge(
        valid_values,
        on=['timestamp', 'neighbor_station_id'],
        how='left',
    )

    mean_cols = [neighbor_cols[feat] for feat in features]
    means = candidates.groupby('_row_id')[mean_cols].mean().reset_index()
    local_cols = {neighbor_cols[feat]: f'_local_{feat}' for feat in features}
    means = means.rename(columns=local_cols)
    out = out.merge(means, on='_row_id', how='left')

    for feat in features:
        local_col = f'_local_{feat}'
        if local_col in out.columns:
            out[feat] = out[feat].where(out[feat].notna(), out[local_col])

    drop_cols = ['_row_id'] + [f'_local_{feat}' for feat in features]
    return out.drop(columns=[col for col in drop_cols if col in out.columns])


def select_output_columns(df):
    mask_cols = [col for col in MASK_COLUMNS if col in df.columns]
    output_cols = [col for col in BASE_OUTPUT_COLUMNS + mask_cols if col in df.columns]
    return df[output_cols]


def build_nbr_map(stations_csv, k_neighbors=K_NEIGHBORS):
    require_gpu_dependencies()
    df = (
        cudf.read_csv(stations_csv, usecols=['station_id', 'latitude', 'longitude'])
        .dropna()
        .drop_duplicates(subset=['station_id'])
        .reset_index(drop=True)
    )
    if len(df) == 0:
        empty = cudf.DataFrame()
        empty['station_id'] = df['station_id'].head(0)
        empty['neighbor_station_id'] = df['station_id'].head(0)
        return empty

    station_ids = df['station_id'].to_pandas().to_numpy()
    coords = df[['latitude', 'longitude']].to_pandas().values.astype('float32')
    n_neighbors = min(k_neighbors + 1, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(cp.asarray(coords))
    nbr_idx = cp.asnumpy(nn.kneighbors(cp.asarray(coords), return_distance=False))

    src_ids = []
    nbr_ids = []
    for row_idx, station_id in enumerate(station_ids):
        added = 0
        for neighbor_idx in nbr_idx[row_idx]:
            neighbor_id = station_ids[int(neighbor_idx)]
            if neighbor_id == station_id:
                continue
            src_ids.append(station_id)
            nbr_ids.append(neighbor_id)
            added += 1
            if added >= k_neighbors:
                break

    if not src_ids:
        empty = cudf.DataFrame()
        empty['station_id'] = df['station_id'].head(0)
        empty['neighbor_station_id'] = df['station_id'].head(0)
        return empty

    return cudf.DataFrame({'station_id': src_ids, 'neighbor_station_id': nbr_ids})


def process_partition(pdf, nbr_map):
    if len(pdf) == 0:
        return select_output_columns(pdf)

    pg = pdf.reset_index(drop=True)
    pg['_partition_order'] = np.arange(len(pg), dtype=np.int64)
    pg = initialize_masked_features(pg)
    pg = apply_partition_local_fill(pg, nbr_map)

    pg['_dir_code'] = pg['direction'].astype('category').cat.codes.astype('int32')
    n_rows = len(pg)
    blocks = (n_rows + THREADS - 1) // THREADS

    valid = pg.loc[~pg['_mask_flag']]
    grp = valid.groupby(['station_id', '_dir_code'])[FEATURES].mean().reset_index()
    if len(grp) > 0:
        sid = pg['station_id'].to_numpy().astype('int32')
        did = pg['_dir_code'].to_numpy().astype('int32')
        mflag = pg['_mask_flag'].to_numpy().astype(bool)
        d_sid = cuda.to_device(sid)
        d_did = cuda.to_device(did)
        d_mask = cuda.to_device(mflag)
        gi = grp[['station_id', '_dir_code']].to_numpy('int32')
        gv = grp[FEATURES].to_numpy('float32')
        d_gi = cuda.to_device(gi)
        d_gv = cuda.to_device(gv)
        for feat_idx, feat in enumerate(FEATURES):
            d_f = cuda.to_device(pg[feat].to_numpy('float32'))
            global_kernel[blocks, THREADS](
                d_f, d_mask, d_sid, d_did, d_gi, d_gv[:, feat_idx], len(grp), d_f
            )
            cuda.synchronize()
            pg[feat] = d_f.copy_to_host()

    pg = pg.sort_values(['station_id', '_dir_code', 'timestamp']).reset_index(drop=True)
    mflag = pg['_mask_flag'].to_numpy().astype(bool)
    ts = pg['timestamp'].astype('int64').to_numpy()
    station_id = pg['station_id'].to_numpy()
    dir_code = pg['_dir_code'].to_numpy()
    changes = (station_id[1:] != station_id[:-1]) | (dir_code[1:] != dir_code[:-1])
    offsets = np.concatenate([[0], np.where(changes)[0] + 1, [n_rows]]).astype('int32')

    d_ts = cuda.to_device(ts)
    d_mask = cuda.to_device(mflag)
    d_offsets = cuda.to_device(offsets)
    for feat in FEATURES:
        d_f = cuda.to_device(pg[feat].to_numpy('float32'))
        temporal_kernel[len(offsets) - 1, THREADS](d_ts, d_mask, d_offsets, d_f)
        cuda.synchronize()
        pg[feat] = d_f.copy_to_host()

    pg = pg.sort_values('_partition_order').reset_index(drop=True)
    pg = pg.drop(columns=['_mask_flag', '_dir_code', '_partition_order'])
    return select_output_columns(pg)


def final_partition_fill(df):
    is_cudf = cudf is not None and isinstance(df, cudf.DataFrame)
    pdf = df.to_pandas() if is_cudf else df.copy()
    if len(pdf) == 0:
        return df

    pdf['_fill_order'] = np.arange(len(pdf), dtype=np.int64)
    ordered = pdf.sort_values(['station_id', 'direction', 'timestamp', '_fill_order'])
    ordered[FEATURES] = (
        ordered.groupby(['station_id', 'direction'], dropna=False)[FEATURES]
        .ffill()
        .bfill()
    )
    ordered = ordered.sort_values('_fill_order').drop(columns=['_fill_order'])

    if is_cudf:
        return cudf.from_pandas(ordered)
    return ordered


def validate_inputs(config):
    for path in (config.raw_parq, config.mask_parq, config.stations_csv):
        if not os.path.exists(path):
            print(f'Missing {path}', file=sys.stderr)
            sys.exit(1)


def main(argv=None):
    config = parse_args(argv)
    require_gpu_dependencies()
    validate_inputs(config)

    raw_schema = parquet_columns(config.raw_parq)
    if raw_schema is not None:
        missing = [col for col in BASE_OUTPUT_COLUMNS if col not in raw_schema]
        if missing:
            print(f'Raw parquet is missing required columns: {missing}', file=sys.stderr)
            sys.exit(1)

    mask_schema = parquet_columns(config.mask_parq)
    mask_key_cols = ['timestamp', 'station_id']
    if mask_schema and 'direction' in mask_schema:
        mask_key_cols.append('direction')
    if mask_schema:
        mask_cols = mask_key_cols + [col for col in MASK_COLUMNS if col in mask_schema]
    else:
        mask_cols = mask_key_cols + MASK_COLUMNS

    with LocalCUDACluster(
        n_workers=config.num_workers,
        CUDA_VISIBLE_DEVICES=config.gpu_devices,
        threads_per_worker=1,
    ) as cluster, Client(cluster) as client:
        nbr_map = build_nbr_map(config.stations_csv)
        scattered_nbr_map = client.scatter(nbr_map, broadcast=True)

        raw = dask_cudf.read_parquet(config.raw_parq, columns=BASE_OUTPUT_COLUMNS)
        mask = dask_cudf.read_parquet(config.mask_parq, columns=mask_cols)
        ddf = raw.merge(mask, on=mask_key_cols, how='left')

        if config.sample_mode:
            npart = max(ddf.npartitions, config.num_workers)
        else:
            npart = max(ddf.npartitions * REPARTITIONS, config.num_workers * 2)
        ddf = ddf.repartition(npartitions=npart)

        meta = process_partition(ddf._meta, nbr_map)
        out = ddf.map_partitions(
            process_partition,
            scattered_nbr_map,
            meta=meta,
            align_dataframes=False,
        ).persist()

        print('Step4: final fill of remaining NaNs')
        out = out.map_partitions(final_partition_fill, meta=out._meta)

        group_means = (
            out[['station_id', 'direction'] + FEATURES]
            .groupby(['station_id', 'direction'])
            .mean()
            .reset_index()
        )
        out = out.merge(group_means, on=['station_id', 'direction'], how='left', suffixes=('', '_gm'))
        for feat in FEATURES:
            out[feat] = out[feat].fillna(out[f'{feat}_gm'])
            out = out.drop(columns=[f'{feat}_gm'])

        global_means = {feat: float(out[feat].mean().compute()) for feat in FEATURES}
        out = out.fillna(global_means)
        out = out[select_output_columns(out._meta).columns]

        out.to_parquet(config.output_dir, write_index=False)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print('Total time:', time.time() - t0, 's')
