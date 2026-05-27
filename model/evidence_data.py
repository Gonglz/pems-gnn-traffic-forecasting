#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data slicing helpers for the PeMS Evidence Pack."""

import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from .evidence_utils import ensure_parent


DEFAULT_SOURCE_DATA_DIR = "/scratch/lgong1/finalproject/pems_data"
DEFAULT_TIME_START = "2025-01-01T00:00:00"
DEFAULT_TIME_END_EXCLUSIVE = "2025-01-08T00:00:00"
FEATURE_ORDER = ["flow_interp", "occupancy_interp", "speed_interp", "tavg", "pcpn", "is_weekend"]


def ensure_evidence_dirs(root):
    root = Path(root)
    for rel in [
        "config",
        "metrics",
        "plots",
        "predictions",
        "logs",
        "checkpoints/single_gpu",
        "checkpoints/ddp_2gpu",
        "pems_data",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def load_graph_nodes(neighbor_path, sids):
    with open(neighbor_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "graph_nodes" in payload:
        return np.asarray(payload["graph_nodes"], dtype=np.int64), payload
    return np.asarray(sids, dtype=np.int64), payload


def derive_timestamps_from_parquet_metadata(parquet_path, expected_len):
    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pyarrow and pandas are required to derive timestamps from parquet metadata") from exc

    pf = pq.ParquetFile(parquet_path)
    names = pf.schema.names
    if "timestamp" not in names:
        raise ValueError(f"{parquet_path} has no timestamp column")
    col_idx = names.index("timestamp")
    mins = []
    maxs = []
    for rg_idx in range(pf.num_row_groups):
        stats = pf.metadata.row_group(rg_idx).column(col_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            mins = []
            break
        mins.append(np.datetime64(stats.min, "ns"))
        maxs.append(np.datetime64(stats.max, "ns"))
    if mins:
        start = min(mins)
        end = max(maxs)
        timestamps = pd.date_range(str(start), str(end), freq="5min").to_numpy(dtype="datetime64[ns]")
        if len(timestamps) == expected_len:
            return timestamps

    table = pf.read(columns=["timestamp"])
    values = np.asarray(table.column("timestamp").to_pandas().unique(), dtype="datetime64[ns]")
    values.sort()
    if len(values) != expected_len:
        raise ValueError(
            f"Derived {len(values)} timestamps from {parquet_path}, expected {expected_len}"
        )
    return values


def load_or_derive_timestamps(source_dir, expected_len):
    source_dir = Path(source_dir)
    timestamps_path = source_dir / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        if len(timestamps) != expected_len:
            raise ValueError(f"{timestamps_path} length {len(timestamps)} != expected {expected_len}")
        return timestamps.astype("datetime64[ns]")

    parquet_path = source_dir / "step51_interpolated_final.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No timestamps.npy and no {parquet_path}; cannot build a reproducible time slice"
        )
    return derive_timestamps_from_parquet_metadata(parquet_path, expected_len)


def select_time_indices(timestamps, time_start=None, time_end_exclusive=None, all_times=False):
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if all_times:
        return np.arange(len(timestamps), dtype=np.int64)
    start = np.datetime64(time_start or DEFAULT_TIME_START, "ns")
    end = np.datetime64(time_end_exclusive or DEFAULT_TIME_END_EXCLUSIVE, "ns")
    mask = (timestamps >= start) & (timestamps < end)
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) == 0:
        raise ValueError(f"No timestamps selected for [{start}, {end})")
    return indices


def copy_if_exists(src, dst):
    src = Path(src)
    if not src.exists():
        return False
    ensure_parent(str(dst))
    shutil.copy2(src, dst)
    return True


def create_data_slice(
    source_data_dir=DEFAULT_SOURCE_DATA_DIR,
    output_root=None,
    time_start=DEFAULT_TIME_START,
    time_end_exclusive=DEFAULT_TIME_END_EXCLUSIVE,
    all_times=False,
    meta_csv=None,
):
    if output_root is None:
        raise ValueError("output_root is required")
    output_root = ensure_evidence_dirs(output_root)
    source = Path(source_data_dir)
    out_data = output_root / "pems_data"
    config_dir = output_root / "config"

    x = np.load(source / "X_ext.npy", mmap_mode="r")
    y = np.load(source / "Y.npy", mmap_mode="r")
    sids = np.asarray(np.load(source / "sids.npy"), dtype=np.int64)
    graph_nodes, _ = load_graph_nodes(source / "step62_neighbors.pkl", sids)
    timestamps = load_or_derive_timestamps(source, expected_len=x.shape[0])

    sid_to_col = {int(sid): idx for idx, sid in enumerate(sids.tolist())}
    missing = [int(sid) for sid in graph_nodes.tolist() if int(sid) not in sid_to_col]
    if missing:
        raise ValueError(f"Graph nodes missing from sids.npy: {missing[:10]}")
    graph_col_indices = np.asarray([sid_to_col[int(sid)] for sid in graph_nodes.tolist()], dtype=np.int64)
    extra_sids = sorted(set(map(int, sids.tolist())) - set(map(int, graph_nodes.tolist())))

    time_indices = select_time_indices(
        timestamps,
        time_start=time_start,
        time_end_exclusive=time_end_exclusive,
        all_times=all_times,
    )
    selected_timestamps = timestamps[time_indices]
    x_slice = np.asarray(x[time_indices][:, graph_col_indices, :], dtype=np.float32)
    y_slice = np.asarray(y[time_indices][:, graph_col_indices], dtype=np.float32)

    np.save(out_data / "X_ext.npy", x_slice)
    np.save(out_data / "Y.npy", y_slice)
    np.save(out_data / "sids.npy", graph_nodes.astype(np.int64))
    np.save(out_data / "timestamps.npy", selected_timestamps.astype("datetime64[ns]"))
    shutil.copy2(source / "step62_neighbors.pkl", out_data / "step62_neighbors.pkl")
    shutil.copy2(source / "step52_edge_index.pt", out_data / "step52_edge_index.pt")
    copy_if_exists(source / "step52_graph_nodes.pkl", out_data / "step52_graph_nodes.pkl")

    resolved_meta = meta_csv or str(source / "step01_d07_meta.csv")
    copied_meta = copy_if_exists(resolved_meta, out_data / "step01_d07_meta.csv")

    payload = {
        "dataset": "PeMS",
        "source_data_dir": str(source),
        "output_dir": str(out_data),
        "time_start": str(selected_timestamps[0]) if len(selected_timestamps) else None,
        "time_end_inclusive": str(selected_timestamps[-1]) if len(selected_timestamps) else None,
        "time_end_exclusive": str(np.datetime64(selected_timestamps[-1], "ns") + np.timedelta64(5, "m"))
        if len(selected_timestamps)
        else None,
        "time_steps": int(len(selected_timestamps)),
        "station_selection": {
            "mode": "all_step62_graph_nodes",
            "station_count": int(len(graph_nodes)),
            "source": "step62_neighbors.pkl:graph_nodes",
        },
        "excluded_non_graph_stations": extra_sids,
        "graph_alignment": "X_ext/Y columns reordered from sids.npy into graph_nodes order; graph files copied unchanged",
        "feature_order": FEATURE_ORDER,
        "target": "flow_interp",
        "train_split": "first 70% of valid time indices",
        "val_split": "next 15% of valid time indices",
        "test_split": "last 15% of valid time indices",
        "array_shapes": {
            "X_ext.npy": list(x_slice.shape),
            "Y.npy": list(y_slice.shape),
            "sids.npy": list(graph_nodes.shape),
            "timestamps.npy": list(selected_timestamps.shape),
        },
        "array_dtypes": {
            "X_ext.npy": str(x_slice.dtype),
            "Y.npy": str(y_slice.dtype),
            "sids.npy": str(graph_nodes.dtype),
            "timestamps.npy": str(selected_timestamps.dtype),
        },
        "meta_csv_copied": bool(copied_meta),
    }
    with open(config_dir / "data_slice.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload
