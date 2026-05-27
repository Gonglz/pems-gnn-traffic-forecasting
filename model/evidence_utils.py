#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for reproducible Evidence Pack metrics."""

import csv
import math
import os
from dataclasses import dataclass

import numpy as np


HORIZONS = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
}
HORIZON_MINUTES = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
}
METRIC_COLUMNS = [
    "run_id",
    "split",
    "model",
    "ablation",
    "horizon_min",
    "mae",
    "rmse",
    "mape",
    "n_samples",
    "train_time_sec",
    "peak_gpu_mem_mb",
    "notes",
]


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def valid_time_indices(total_times, max_horizon=max(HORIZONS.values())):
    if total_times <= max_horizon:
        raise ValueError(f"Need more than {max_horizon} time steps, got {total_times}")
    return np.arange(total_times - max_horizon, dtype=np.int64)


def split_time_indices(total_times, train_frac=0.70, val_frac=0.15, max_horizon=max(HORIZONS.values())):
    valid = valid_time_indices(total_times, max_horizon=max_horizon)
    n = len(valid)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if n >= 3:
        train_end = min(max(train_end, 1), n - 2)
        val_end = min(max(val_end, train_end + 1), n - 1)
    return {
        "train": valid[:train_end],
        "val": valid[train_end:val_end],
        "test": valid[val_end:],
    }


@dataclass
class MetricAccumulator:
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_mape: float = 0.0
    n: int = 0

    def update(self, pred, true, eps=1.0):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        err = pred - true
        denom = np.maximum(np.abs(true), eps)
        self.sum_abs += float(np.abs(err).sum())
        self.sum_sq += float((err ** 2).sum())
        self.sum_mape += float((np.abs(err) / denom).sum() * 100.0)
        self.n += int(err.size)

    def row(self, run_id, split, model, ablation, horizon_min, train_time_sec="", peak_gpu_mem_mb="", notes=""):
        if self.n == 0:
            mae = rmse = mape = math.nan
        else:
            mae = self.sum_abs / self.n
            rmse = math.sqrt(self.sum_sq / self.n)
            mape = self.sum_mape / self.n
        return {
            "run_id": run_id,
            "split": split,
            "model": model,
            "ablation": ablation,
            "horizon_min": horizon_min,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "n_samples": self.n,
            "train_time_sec": train_time_sec,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
            "notes": notes,
        }


def compute_metric_row(pred, true, run_id, split, model, ablation, horizon_min, notes="", eps=1.0):
    acc = MetricAccumulator()
    acc.update(pred, true, eps=eps)
    return acc.row(run_id, split, model, ablation, horizon_min, notes=notes)


def write_metric_rows(path, rows):
    ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in METRIC_COLUMNS})


def append_metric_rows(path, rows):
    ensure_parent(path)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in METRIC_COLUMNS})


def read_metric_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def timestamp_slots(timestamps):
    values = np.asarray(timestamps)
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    dt = values.astype("datetime64[m]")
    day = dt.astype("datetime64[D]")
    minutes = (dt - day).astype("timedelta64[m]").astype(np.int64)
    return (minutes // 5).astype(np.int64)


def timestamp_hours(timestamps):
    slots = timestamp_slots(timestamps)
    return (slots // 12).astype(np.int64)


def write_csv(path, fieldnames, rows):
    ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

