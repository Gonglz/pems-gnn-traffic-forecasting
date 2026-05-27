#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 model-quality helpers: task audit, temporal features, and normalization."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from .evidence_utils import HORIZON_MINUTES, HORIZONS, timestamp_slots


BASE_FEATURE_ORDER = [
    "flow_interp",
    "occupancy_interp",
    "speed_interp",
    "tavg",
    "pcpn",
    "is_weekend",
]
TEMPORAL_FEATURE_ORDER = [
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_week",
    "cos_day_of_week",
    "is_holiday",
]
NORMALIZATION_NONE = "none"
NORMALIZATION_STATION_WISE_FLOW = "station-wise-flow"
NORMALIZATION_CHOICES = (NORMALIZATION_NONE, NORMALIZATION_STATION_WISE_FLOW)
MODEL_TYPE_CHOICES = ("graphsage", "mlp")


def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(day):
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def us_federal_holidays(year):
    fixed = [
        date(year, 1, 1),
        date(year, 6, 19),
        date(year, 7, 4),
        date(year, 11, 11),
        date(year, 12, 25),
    ]
    floating = [
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _nth_weekday(year, 11, 3, 4),
    ]
    holidays = set(fixed + floating)
    holidays.update(_observed(day) for day in fixed)
    return holidays


def _to_python_dates(timestamps):
    days = np.asarray(timestamps).astype("datetime64[D]").astype(np.int64)
    epoch = date(1970, 1, 1)
    return [epoch + timedelta(days=int(day)) for day in days]


def temporal_feature_matrix(timestamps):
    timestamps = np.asarray(timestamps)
    if timestamps.size == 0:
        return np.zeros((0, len(TEMPORAL_FEATURE_ORDER)), dtype=np.float32)

    slots = timestamp_slots(timestamps).astype(np.float32)
    tod_angle = 2.0 * np.pi * slots / 288.0
    days = timestamps.astype("datetime64[D]")
    weekdays = ((days.astype(np.int64) + 3) % 7).astype(np.float32)
    dow_angle = 2.0 * np.pi * weekdays / 7.0

    py_dates = _to_python_dates(timestamps)
    holiday_cache = {}
    is_holiday = []
    for day in py_dates:
        if day.year not in holiday_cache:
            holiday_cache[day.year] = us_federal_holidays(day.year)
        is_holiday.append(1.0 if day in holiday_cache[day.year] else 0.0)

    return np.column_stack(
        [
            np.sin(tod_angle),
            np.cos(tod_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            np.asarray(is_holiday, dtype=np.float32),
        ]
    ).astype(np.float32)


def apply_temporal_encoding_(dataset, enabled=True):
    feature_order = list(getattr(dataset, "feature_order", BASE_FEATURE_ORDER[: dataset.F]))
    if not enabled:
        dataset.feature_order = feature_order
        return {"enabled": False, "added_features": [], "feature_order": feature_order}
    if getattr(dataset, "_temporal_encoding_applied", False):
        return {
            "enabled": True,
            "added_features": TEMPORAL_FEATURE_ORDER,
            "feature_order": list(dataset.feature_order),
        }
    if dataset.timestamps is None:
        raise ValueError("Temporal encoding requires timestamps.npy")

    feats = torch.from_numpy(temporal_feature_matrix(dataset.timestamps)).to(dataset.X.dtype)
    expanded = feats[:, None, :].expand(dataset.T, dataset.N, feats.shape[1])
    dataset.X = torch.cat([dataset.X, expanded], dim=2).contiguous()
    dataset.X_ext = dataset.X
    dataset.F = int(dataset.X.shape[2])
    dataset.feature_order = feature_order + TEMPORAL_FEATURE_ORDER
    dataset._temporal_encoding_applied = True
    return {
        "enabled": True,
        "added_features": TEMPORAL_FEATURE_ORDER,
        "feature_order": list(dataset.feature_order),
    }


def audit_forecast_task(dataset, train_times=None, max_samples=16):
    if dataset.X.ndim != 3 or dataset.Y.ndim != 2:
        raise ValueError("Expected X shape [T, N, F] and Y shape [T, N]")
    if dataset.X.shape[0] != dataset.Y.shape[0] or dataset.X.shape[1] != dataset.Y.shape[1]:
        raise ValueError(f"X/Y time-node shape mismatch: X={tuple(dataset.X.shape)}, Y={tuple(dataset.Y.shape)}")
    if dataset.T <= max(HORIZONS.values()):
        raise ValueError(f"Need more than {max(HORIZONS.values())} timestamps for P2 horizons")

    source_times = np.arange(dataset.T - max(HORIZONS.values()), dtype=np.int64)
    if train_times is not None and len(train_times) > 0:
        source_times = np.asarray(train_times, dtype=np.int64)
    sampled = source_times[: max(1, min(len(source_times), int(max_samples)))]

    x_flow = dataset.X[sampled, :, 0].detach().cpu()
    y_current = dataset.Y[sampled, :].detach().cpu()
    current_exact = torch.isclose(x_flow, y_current, rtol=1e-4, atol=1e-4).float().mean().item()
    current_mae = torch.mean(torch.abs(x_flow - y_current)).item()

    future_checks = {}
    for head, delta in HORIZONS.items():
        y_future = dataset.Y[sampled + delta, :].detach().cpu()
        exact = torch.isclose(x_flow, y_future, rtol=1e-4, atol=1e-4).float().mean().item()
        mae = torch.mean(torch.abs(x_flow - y_future)).item()
        future_checks[head] = {
            "delta_steps": int(delta),
            "horizon_min": int(HORIZON_MINUTES[head]),
            "x_flow_equals_future_target_rate": float(exact),
            "x_flow_to_future_target_mae": float(mae),
        }
        if exact > 0.999 and current_exact < 0.999:
            raise ValueError(
                f"Potential target leakage: X[t, flow] exactly matches Y[t+{delta}] "
                f"for horizon {head}, but does not match Y[t]."
            )

    return {
        "target": "flow_interp",
        "input_timestamp": "X[t] current timestamp features",
        "target_indices": {
            head: f"Y[t+{delta}] ({HORIZON_MINUTES[head]} minutes)"
            for head, delta in HORIZONS.items()
        },
        "horizons": {
            head: {"delta_steps": int(delta), "minutes": int(HORIZON_MINUTES[head])}
            for head, delta in HORIZONS.items()
        },
        "sampled_source_times": int(len(sampled)),
        "x_flow_equals_current_target_rate": float(current_exact),
        "x_flow_to_current_target_mae": float(current_mae),
        "future_leakage_checks": future_checks,
        "status": "passed",
    }


def compute_stationwise_flow_stats(X, Y, train_times, head_deltas=HORIZONS, eps=1e-6):
    if len(train_times) == 0:
        raise ValueError("Cannot compute normalization stats with no train times")
    train_times = torch.as_tensor(train_times, dtype=torch.long)
    input_values = X[train_times, :, 0].float()
    input_mean = input_values.mean(dim=0)
    input_std = input_values.std(dim=0, unbiased=False).clamp_min(float(eps))

    target_mean = {}
    target_std = {}
    for head, delta in head_deltas.items():
        target_idx = train_times + int(delta)
        values = Y[target_idx, :].float()
        target_mean[head] = values.mean(dim=0)
        target_std[head] = values.std(dim=0, unbiased=False).clamp_min(float(eps))

    return {
        "mode": NORMALIZATION_STATION_WISE_FLOW,
        "input_flow_mean": input_mean,
        "input_flow_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def apply_input_flow_normalization_(X, stats):
    if not stats or stats.get("mode") == NORMALIZATION_NONE:
        return
    mean = stats["input_flow_mean"].to(dtype=X.dtype)
    std = stats["input_flow_std"].to(dtype=X.dtype)
    X[:, :, 0] = (X[:, :, 0] - mean[None, :]) / std[None, :]


def normalize_target(y, stats, head, seed_nodes, device):
    if not stats or stats.get("mode") == NORMALIZATION_NONE:
        return y
    seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long, device="cpu")
    mean = stats["target_mean"][head][seed_nodes].to(device=device, dtype=y.dtype).unsqueeze(-1)
    std = stats["target_std"][head][seed_nodes].to(device=device, dtype=y.dtype).unsqueeze(-1)
    return (y - mean) / std


def inverse_transform_prediction(pred, stats, head, seed_nodes):
    if not stats or stats.get("mode") == NORMALIZATION_NONE:
        return pred
    device = pred.device
    seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long, device="cpu")
    mean = stats["target_mean"][head][seed_nodes].to(device=device, dtype=pred.dtype).unsqueeze(-1)
    std = stats["target_std"][head][seed_nodes].to(device=device, dtype=pred.dtype).unsqueeze(-1)
    return pred * std + mean


def save_normalization_stats(path, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input_flow_mean": stats["input_flow_mean"].detach().cpu().numpy(),
        "input_flow_std": stats["input_flow_std"].detach().cpu().numpy(),
    }
    for head in HORIZONS:
        arrays[f"target_mean_{head}"] = stats["target_mean"][head].detach().cpu().numpy()
        arrays[f"target_std_{head}"] = stats["target_std"][head].detach().cpu().numpy()
    np.savez(path, **arrays)
    meta = {
        "mode": stats.get("mode", NORMALIZATION_STATION_WISE_FLOW),
        "target": "flow_interp",
        "horizons": {
            head: {"delta_steps": int(delta), "minutes": int(HORIZON_MINUTES[head])}
            for head, delta in HORIZONS.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_normalization_stats(path):
    path = Path(path)
    loaded = np.load(path)
    stats = {
        "mode": NORMALIZATION_STATION_WISE_FLOW,
        "input_flow_mean": torch.from_numpy(loaded["input_flow_mean"]).float(),
        "input_flow_std": torch.from_numpy(loaded["input_flow_std"]).float(),
        "target_mean": {},
        "target_std": {},
    }
    for head in HORIZONS:
        stats["target_mean"][head] = torch.from_numpy(loaded[f"target_mean_{head}"]).float()
        stats["target_std"][head] = torch.from_numpy(loaded[f"target_std_{head}"]).float()
    return stats


def normalization_summary(stats, path=None):
    if not stats or stats.get("mode") == NORMALIZATION_NONE:
        return {"mode": NORMALIZATION_NONE}
    return {
        "mode": stats["mode"],
        "scope": "station-wise train source times for input flow; station-wise train target times per horizon",
        "stats_path": str(path) if path else "",
        "input_flow_mean_mean": float(stats["input_flow_mean"].mean().item()),
        "input_flow_std_mean": float(stats["input_flow_std"].mean().item()),
    }
