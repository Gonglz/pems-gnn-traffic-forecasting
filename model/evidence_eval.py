#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline, checkpoint, and plot generation for Evidence Packs."""

import json
import math
import os
import pickle
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evidence_utils import (
    HORIZON_MINUTES,
    HORIZONS,
    MetricAccumulator,
    ensure_parent,
    read_metric_rows,
    split_time_indices,
    timestamp_hours,
    timestamp_slots,
    write_metric_rows,
)
from .p2_quality import (
    NORMALIZATION_STATION_WISE_FLOW,
    apply_input_flow_normalization_,
    apply_temporal_encoding_,
    inverse_transform_prediction,
    load_normalization_stats,
)
from .training_modes import ablation_label, apply_feature_ablation_, apply_graph_mode


def load_arrays(data_dir):
    data_dir = Path(data_dir)
    return {
        "X": np.load(data_dir / "X_ext.npy", mmap_mode="r"),
        "Y": np.load(data_dir / "Y.npy", mmap_mode="r"),
        "sids": np.asarray(np.load(data_dir / "sids.npy"), dtype=np.int64),
        "timestamps": np.load(data_dir / "timestamps.npy"),
    }


def add_all_row(rows, run_id, split, model, ablation, accumulators, train_time_sec="", peak_gpu_mem_mb="", notes=""):
    all_acc = MetricAccumulator()
    for acc in accumulators.values():
        all_acc.sum_abs += acc.sum_abs
        all_acc.sum_sq += acc.sum_sq
        all_acc.sum_mape += acc.sum_mape
        all_acc.n += acc.n
    rows.append(
        all_acc.row(
            run_id,
            split,
            model,
            ablation,
            "all",
            train_time_sec=train_time_sec,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            notes=notes,
        )
    )


def last_value_baseline(Y, test_times):
    accs = {name: MetricAccumulator() for name in HORIZONS}
    for name, delta in HORIZONS.items():
        for time_idx in test_times:
            accs[name].update(Y[time_idx], Y[time_idx + delta])
    return accs


def historical_average_baseline(Y, timestamps, train_times, test_times):
    slots = timestamp_slots(timestamps)
    accs = {name: MetricAccumulator() for name in HORIZONS}
    station_count = Y.shape[1]
    for name, delta in HORIZONS.items():
        sums = np.zeros((288, station_count), dtype=np.float64)
        counts = np.zeros((288, station_count), dtype=np.float64)
        train_targets = []
        for time_idx in train_times:
            target_idx = int(time_idx + delta)
            slot = int(slots[target_idx])
            values = np.asarray(Y[target_idx], dtype=np.float64)
            sums[slot] += values
            counts[slot] += 1.0
            train_targets.append(values)
        station_mean = np.mean(np.stack(train_targets, axis=0), axis=0)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        for time_idx in test_times:
            target_idx = int(time_idx + delta)
            slot = int(slots[target_idx])
            pred = np.where(counts[slot] > 0, means[slot], station_mean)
            accs[name].update(pred, Y[target_idx])
    return accs


def _sample_flat_indices(num_times, num_nodes, max_samples, rng):
    total = int(num_times) * int(num_nodes)
    limit = total if max_samples is None or max_samples <= 0 else min(total, int(max_samples))
    if limit == total:
        flat = np.arange(total, dtype=np.int64)
    else:
        flat = rng.choice(total, size=limit, replace=False).astype(np.int64)
    return flat // num_nodes, flat % num_nodes


def _rf_features(X, timestamps, times, nodes, horizon_min):
    base = np.asarray(X[times, nodes, :], dtype=np.float32)
    ts = np.asarray(timestamps)[times]
    slots = timestamp_slots(ts).astype(np.float32) / 287.0
    days = ts.astype("datetime64[D]")
    weekday = ((days.astype(int) + 3) % 7).astype(np.float32) / 6.0
    node_norm = nodes.astype(np.float32) / max(float(X.shape[1] - 1), 1.0)
    extra = np.column_stack(
        [
            np.full(len(times), float(horizon_min) / 30.0, dtype=np.float32),
            slots,
            weekday,
            node_norm,
        ]
    )
    return np.concatenate([base, extra], axis=1)


def random_forest_baseline(
    X,
    Y,
    timestamps,
    train_times,
    test_times,
    max_train_samples=100000,
    max_test_samples=100000,
    seed=42,
):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the RandomForest baseline") from exc

    rng = np.random.default_rng(seed)
    accs = {name: MetricAccumulator() for name in HORIZONS}
    notes = []
    for name, delta in HORIZONS.items():
        train_pos, train_nodes = _sample_flat_indices(len(train_times), Y.shape[1], max_train_samples, rng)
        test_pos, test_nodes = _sample_flat_indices(len(test_times), Y.shape[1], max_test_samples, rng)
        train_t = np.asarray(train_times, dtype=np.int64)[train_pos]
        test_t = np.asarray(test_times, dtype=np.int64)[test_pos]
        train_X = _rf_features(X, timestamps, train_t, train_nodes, HORIZON_MINUTES[name])
        train_y = np.asarray(Y[train_t + delta, train_nodes], dtype=np.float32)
        test_X = _rf_features(X, timestamps, test_t, test_nodes, HORIZON_MINUTES[name])
        test_y = np.asarray(Y[test_t + delta, test_nodes], dtype=np.float32)
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        accs[name].update(pred, test_y)
        notes.append(f"{name}:train={len(train_y)},test={len(test_y)}")
    return accs, ";".join(notes)


def baseline_metric_rows(
    data_dir,
    run_id,
    max_rf_train_samples=100000,
    max_rf_test_samples=100000,
    seed=42,
    include_rf=True,
):
    arrays = load_arrays(data_dir)
    X, Y, timestamps = arrays["X"], arrays["Y"], arrays["timestamps"]
    splits = split_time_indices(Y.shape[0])
    rows = []

    for model_name, accs, notes in [
        ("last_value", last_value_baseline(Y, splits["test"]), "naive temporal"),
        (
            "historical_average",
            historical_average_baseline(Y, timestamps, splits["train"], splits["test"]),
            "time-of-day average from train split",
        ),
    ]:
        for name, acc in accs.items():
            rows.append(acc.row(run_id, "test", model_name, "none", HORIZON_MINUTES[name], notes=notes))
        add_all_row(rows, run_id, "test", model_name, "none", accs, notes=notes)

    if include_rf:
        rf_accs, rf_notes = random_forest_baseline(
            X,
            Y,
            timestamps,
            splits["train"],
            splits["test"],
            max_train_samples=max_rf_train_samples,
            max_test_samples=max_rf_test_samples,
            seed=seed,
        )
        for name, acc in rf_accs.items():
            rows.append(acc.row(run_id, "test", "random_forest", "none", HORIZON_MINUTES[name], notes=rf_notes))
        add_all_row(rows, run_id, "test", "random_forest", "none", rf_accs, notes=rf_notes)
    return rows


def write_baseline_metrics(data_dir, output_csv, run_id, **kwargs):
    rows = baseline_metric_rows(data_dir, run_id, **kwargs)
    write_metric_rows(output_csv, rows)
    return rows


def evaluate_checkpoint_metrics(
    data_dir,
    edge_index_path,
    checkpoint_path,
    run_id,
    output_csv=None,
    batch_size=1024,
    max_eval_times=0,
    device=None,
    plot_station_id=None,
    prediction_plot_path=None,
    prediction_csv_path=None,
    error_heatmap_path=None,
    model_name="graphsage_weather",
    ablation="full",
    feature_ablation="none",
    graph_mode="full",
    feature_ablation_values=None,
    model_type="graphsage",
    normalization="none",
    normalization_stats_path="",
    temporal_encoding=False,
    station_embedding_dim=0,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3,
    train_time_sec="",
    peak_gpu_mem_mb="",
    notes="",
):
    import torch
    from torch_geometric.loader import NeighborSampler

    from .dataset_full import RFGraphDatasetFull
    from .gnn_final import MultiHeadRFGraphSAGEDyn, MultiHeadRFMLP
    from .predict_and_plot import load_checkpoint

    ds = RFGraphDatasetFull(data_dir=data_dir)
    splits = split_time_indices(ds.T)
    apply_feature_ablation_(
        ds.X,
        splits["train"],
        feature_ablation,
        values=feature_ablation_values,
    )
    temporal_metadata = apply_temporal_encoding_(ds, enabled=temporal_encoding)
    normalization_stats = None
    if normalization == NORMALIZATION_STATION_WISE_FLOW:
        stats_path = Path(normalization_stats_path) if normalization_stats_path else Path(checkpoint_path).parent / "normalization_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing normalization stats for checkpoint eval: {stats_path}")
        normalization_stats = load_normalization_stats(stats_path)
        apply_input_flow_normalization_(ds.X, normalization_stats)
    if ablation == "full":
        ablation = ablation_label(feature_ablation, graph_mode)
    test_times = list(map(int, splits["test"]))
    if max_eval_times and max_eval_times > 0:
        test_times = test_times[: int(max_eval_times)]
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    edge_index = None
    if model_type != "mlp":
        edge_index = torch.load(edge_index_path, map_location="cpu").long().contiguous()
        edge_index = apply_graph_mode(edge_index, ds.N, graph_mode)
        model = MultiHeadRFGraphSAGEDyn(
            in_dim=ds.F,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_nodes=ds.N,
            station_embedding_dim=station_embedding_dim,
        ).to(device)
    else:
        model = MultiHeadRFMLP(
            in_dim=ds.F,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_nodes=ds.N,
            station_embedding_dim=station_embedding_dim,
        ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    accs = {name: MetricAccumulator() for name in HORIZONS}
    hours_abs = {name: np.zeros(24, dtype=np.float64) for name in HORIZONS}
    hours_n = {name: np.zeros(24, dtype=np.float64) for name in HORIZONS}
    station_idx = 0
    if plot_station_id is not None:
        station_idx = ds.station_id_to_node_idx(plot_station_id)
    station_records = {name: {"time": [], "pred": [], "true": []} for name in HORIZONS}

    samplers = {}
    if model_type != "mlp":
        samplers = {
            name: NeighborSampler(
                edge_index,
                sizes=sizes,
                node_idx=torch.arange(ds.N),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            for name, sizes in {"5min": [8, 8], "15min": [12, 12], "30min": [16, 16]}.items()
        }

    with torch.no_grad():
        for time_idx in test_times:
            for name, delta in HORIZONS.items():
                if model_type == "mlp":
                    batch_iter = []
                    for start in range(0, ds.N, batch_size):
                        seed_nodes = torch.arange(start, min(ds.N, start + batch_size), dtype=torch.long)
                        batch_iter.append((len(seed_nodes), seed_nodes, None))
                else:
                    batch_iter = samplers[name]
                for batch_size_actual, node_ids, adjs in batch_iter:
                    x = ds.X[time_idx, node_ids].to(device)
                    if model_type == "mlp":
                        pred_t = model(x=x, head=name, node_ids=node_ids.to(device))[:batch_size_actual]
                    else:
                        adjs = [adj.to(device) for adj in adjs]
                        pred_t = model(
                            x,
                            adjs,
                            head=name,
                            node_ids=node_ids.to(device, non_blocking=True),
                        )[:batch_size_actual]
                    seed_nodes_t = node_ids[:batch_size_actual]
                    pred_t = inverse_transform_prediction(
                        pred_t,
                        normalization_stats,
                        name,
                        seed_nodes_t,
                    )
                    pred = pred_t.squeeze(1).detach().cpu().numpy()
                    seed_nodes = seed_nodes_t.cpu().numpy()
                    true = ds.Y[time_idx + delta, seed_nodes].cpu().numpy()
                    accs[name].update(pred, true)
                    hour = int(timestamp_hours(ds.timestamps[[time_idx + delta]])[0]) if ds.timestamps is not None else 0
                    hours_abs[name][hour] += float(np.abs(pred - true).sum())
                    hours_n[name][hour] += len(true)
                    match = np.flatnonzero(seed_nodes == station_idx)
                    if match.size:
                        idx = int(match[0])
                        station_records[name]["time"].append(ds.idx_to_time(time_idx + delta))
                        station_records[name]["pred"].append(float(pred[idx]))
                        station_records[name]["true"].append(float(true[idx]))

    rows = []
    for name, acc in accs.items():
        rows.append(
            acc.row(
                run_id,
                "test",
                model_name,
                ablation,
                HORIZON_MINUTES[name],
                train_time_sec=train_time_sec,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                notes=notes,
            )
        )
    add_all_row(
        rows,
        run_id,
        "test",
        model_name,
        ablation,
        accs,
        train_time_sec=train_time_sec,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        notes=notes,
    )
    if output_csv:
        write_metric_rows(output_csv, rows)

    if prediction_plot_path:
        plot_prediction_vs_truth(prediction_plot_path, ds.node_idx_to_station_id(station_idx), station_records)
    if prediction_csv_path:
        write_prediction_samples(prediction_csv_path, ds.node_idx_to_station_id(station_idx), station_records)
    if error_heatmap_path:
        plot_error_heatmap(error_heatmap_path, hours_abs, hours_n)
    return rows


def write_prediction_samples(path, station_id, station_records):
    ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["station_id", "horizon", "timestamp", "prediction", "truth", "abs_error"],
        )
        writer.writeheader()
        for horizon, record in station_records.items():
            for timestamp, pred, true in zip(record["time"], record["pred"], record["true"]):
                writer.writerow(
                    {
                        "station_id": int(station_id),
                        "horizon": horizon,
                        "timestamp": str(timestamp),
                        "prediction": float(pred),
                        "truth": float(true),
                        "abs_error": abs(float(pred) - float(true)),
                    }
                )


def plot_prediction_vs_truth(path, station_id, station_records):
    ensure_parent(path)
    plt.figure(figsize=(10, 4))
    for name, record in station_records.items():
        if not record["time"]:
            continue
        label = f"{name} pred"
        plt.plot(record["time"], record["pred"], label=label, linewidth=1.3)
    first = next((record for record in station_records.values() if record["time"]), None)
    if first:
        plt.plot(first["time"], first["true"], label="truth", color="black", linewidth=1.4)
    plt.title(f"Prediction vs truth for station {station_id}")
    plt.xlabel("time")
    plt.ylabel("flow")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_error_heatmap(path, hours_abs, hours_n):
    ensure_parent(path)
    labels = list(HORIZONS.keys())
    mat = []
    for name in labels:
        mat.append(np.divide(hours_abs[name], hours_n[name], out=np.zeros(24), where=hours_n[name] > 0))
    mat = np.asarray(mat)
    plt.figure(figsize=(10, 3))
    plt.imshow(mat, aspect="auto", cmap="magma")
    plt.colorbar(label="mean absolute error")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(0, 24, 2), [str(x) for x in range(0, 24, 2)])
    plt.xlabel("hour of day")
    plt.title("Forecast error by horizon and hour")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_topology(data_dir, meta_csv, output_path, max_edges=30000):
    import torch

    data_dir = Path(data_dir)
    with open(data_dir / "step62_neighbors.pkl", "rb") as f:
        payload = pickle.load(f)
    sids = np.asarray(payload.get("graph_nodes", np.load(data_dir / "sids.npy")), dtype=np.int64)
    meta = pd.read_csv(meta_csv)
    if "ID" in meta.columns and "station_id" not in meta.columns:
        meta = meta.rename(columns={"ID": "station_id"})
    meta["station_id"] = meta["station_id"].astype(int)
    meta = meta.drop_duplicates("station_id").set_index("station_id")
    coords = meta.reindex(sids)[["longitude", "latitude"]]
    valid = coords.notna().all(axis=1).to_numpy()
    edge_index = torch.load(data_dir / "step52_edge_index.pt", map_location="cpu").numpy()
    if edge_index.shape[1] > max_edges:
        edge_index = edge_index[:, :max_edges]

    ensure_parent(output_path)
    plt.figure(figsize=(7, 7))
    xy = coords.to_numpy(dtype=float)
    for src, dst in edge_index.T:
        if src < len(xy) and dst < len(xy) and valid[src] and valid[dst]:
            plt.plot([xy[src, 0], xy[dst, 0]], [xy[src, 1], xy[dst, 1]], color="#888888", alpha=0.08, linewidth=0.4)
    plt.scatter(xy[valid, 0], xy[valid, 1], s=5, c="#1f77b4", alpha=0.75)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("PeMS sensor topology")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_ablation_metrics(run_root):
    run_root = Path(run_root)
    model_path = run_root / "metrics" / "model_metrics.csv"
    baseline_path = run_root / "metrics" / "baseline_metrics.csv"
    rows = []
    if model_path.exists():
        for row in read_metric_rows(model_path):
            rows.append(dict(row))
    if baseline_path.exists():
        for row in read_metric_rows(baseline_path):
            if row.get("model") == "random_forest":
                copied = dict(row)
                copied["ablation"] = "without_graph_neighbors"
                copied["notes"] = (copied.get("notes", "") + ";non_graph_rf_proxy").strip(";")
                rows.append(copied)
    run_id = run_root.name
    existing_ablations = {row.get("ablation") for row in rows}
    missing_ablations = [
        ablation
        for ablation in ["without_graph_neighbors", "without_weather", "simple_ffill"]
        if ablation not in existing_ablations
    ]
    for ablation in missing_ablations:
        for horizon in [5, 15, 30, "all"]:
            rows.append(
                {
                    "run_id": run_id,
                    "split": "test",
                    "model": "graphsage_weather",
                    "ablation": ablation,
                    "horizon_min": horizon,
                    "mae": "",
                    "rmse": "",
                    "mape": "",
                    "n_samples": 0,
                    "train_time_sec": "",
                    "peak_gpu_mem_mb": "",
                    "notes": "not_available",
                }
            )
    write_metric_rows(run_root / "metrics" / "ablation_metrics.csv", rows)
    return rows


def write_scaling_metrics(run_root):
    import csv

    run_root = Path(run_root)
    rows = []
    for setup, checkpoint_dir in [
        ("1 GPU", run_root / "checkpoints" / "single_gpu"),
        ("2 GPU DDP", run_root / "checkpoints" / "ddp_2gpu"),
    ]:
        metrics_path = checkpoint_dir / "epoch_metrics.csv"
        config_path = checkpoint_dir / "train_config.json"
        log_path = run_root / "logs" / ("single_gpu.log" if setup == "1 GPU" else "ddp_2gpu.log")
        total = peak = ""
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            total = config.get("total_time_sec", "")
            peak = config.get("peak_gpu_mem_mb", "")
        epoch_time = ""
        if metrics_path.exists():
            with open(metrics_path, newline="") as f:
                metric_rows = list(csv.DictReader(f))
            train_rows = [r for r in metric_rows if r.get("phase") == "train"]
            if train_rows:
                vals = [float(r["phase_time_sec"]) for r in train_rows if r.get("phase_time_sec")]
                epoch_time = sum(vals) / len(vals) if vals else ""
        skipped = False
        if log_path.exists():
            skipped = "skipped" in log_path.read_text(errors="ignore").lower()
        if skipped:
            notes = "skipped_by_config"
        elif setup == "1 GPU":
            notes = "baseline"
        else:
            notes = f"ddp_shard_mode={config.get('ddp_shard_mode', 'unknown')}"
        rows.append(
            {
                "setup": setup,
                "epoch_time_sec": epoch_time,
                "speedup": "",
                "peak_gpu_mem_mb": peak,
                "total_time_sec": total,
                "notes": notes,
            }
        )
    if rows and rows[0]["epoch_time_sec"]:
        base = float(rows[0]["epoch_time_sec"])
        rows[0]["speedup"] = 1.0
        if len(rows) > 1 and rows[1]["epoch_time_sec"]:
            rows[1]["speedup"] = base / float(rows[1]["epoch_time_sec"])
    out = run_root / "metrics" / "scaling_metrics.csv"
    ensure_parent(out)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setup", "epoch_time_sec", "speedup", "peak_gpu_mem_mb", "total_time_sec", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows
