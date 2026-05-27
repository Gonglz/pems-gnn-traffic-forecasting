#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI orchestration for PeMS Evidence Packs."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .evidence_data import (
    DEFAULT_SOURCE_DATA_DIR,
    DEFAULT_TIME_END_EXCLUSIVE,
    DEFAULT_TIME_START,
    create_data_slice,
    ensure_evidence_dirs,
)
from .evidence_eval import (
    evaluate_checkpoint_metrics,
    plot_topology,
    write_ablation_metrics,
    write_baseline_metrics,
    write_scaling_metrics,
)
from .evidence_utils import write_metric_rows
from .training_modes import ablation_label


DEFAULT_ROOT_BASE = "/scratch2/lgong1/finalproject_gpu_parallel_test"
DEFAULT_PYTHON = "/scratch/lgong1/envs/traffic-env/bin/python"
DEFAULT_TORCHRUN = "/scratch/lgong1/envs/traffic-env/bin/torchrun"
DEFAULT_BACKUP_GLOB = "/scratch2/lgong1/finalproject_backups/*evidence-code-preupload.tar.gz"


def timestamp_id():
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_output_root(output_root=None, root_base=DEFAULT_ROOT_BASE):
    if output_root:
        return Path(output_root)
    return Path(root_base) / timestamp_id()


def write_train_config(run_root, args):
    config = {
        "run_id": Path(run_root).name,
        "data_dir": str(Path(run_root) / "pems_data"),
        "edge_index_path": str(Path(run_root) / "pems_data" / "step52_edge_index.pt"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "time_stride": args.time_stride,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "val_period": args.val_period,
        "max_train_times": args.max_train_times,
        "max_val_times": args.max_val_times,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "accum_steps": args.accum_steps,
        "early_stop_patience": args.early_stop_patience,
        "amp": not args.no_amp,
        "ddp_shard_mode": args.ddp_shard_mode,
        "feature_ablation": args.feature_ablation,
        "graph_mode": args.graph_mode,
        "normalization": args.normalization,
        "temporal_encoding": not args.no_temporal_encoding,
        "station_embedding_dim": args.station_embedding_dim,
        "model_type": args.model_type,
        "skip_single": args.skip_single,
        "single_gpu": args.single_gpu,
        "ddp_gpus": args.ddp_gpus,
        "code_ref": resolve_code_ref(args.code_ref),
    }
    path = Path(run_root) / "config" / "train_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def resolve_code_ref(code_ref=None):
    if code_ref:
        return str(code_ref)
    candidates = sorted(Path("/").glob(DEFAULT_BACKUP_GLOB.lstrip("/")), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else "not_available"


def _capture(command, timeout=30):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=timeout).strip()
    except Exception as exc:
        return f"not_available: {exc}"


def write_run_metadata(run_root, args):
    config_dir = Path(run_root) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    keep = ("torch", "geometric", "numpy", "pandas", "cudf", "scipy", "sklearn", "scikit", "matplotlib")
    lines = [
        f"python_bin={args.python_bin}",
        _capture([args.python_bin, "-V"]),
        "",
        "[selected pip freeze]",
    ]
    freeze = _capture([args.python_bin, "-m", "pip", "freeze"], timeout=60)
    for line in freeze.splitlines():
        if any(token in line.lower() for token in keep):
            lines.append(line)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        lines.extend(
            [
                "",
                "[nvidia-smi]",
                _capture(
                    [
                        nvidia_smi,
                        "--query-gpu=index,name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    timeout=20,
                ),
            ]
        )
    (config_dir / "environment.txt").write_text("\n".join(lines).rstrip() + "\n")
    (config_dir / "git_or_backup_ref.txt").write_text(resolve_code_ref(args.code_ref) + "\n")


def _checkpoint_train_config(checkpoint_dir):
    config_path = Path(checkpoint_dir) / "train_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def evaluate_available_checkpoints(args, data_dir, run_root):
    if args.checkpoint_path:
        checkpoint_specs = [("graphsage_weather_external", Path(args.checkpoint_path), "external checkpoint")]
    else:
        checkpoint_specs = [
            (
                "graphsage_weather_single_gpu",
                run_root / "checkpoints" / "single_gpu" / "best_rf_gnn_dynamic_sampler.pth",
                "single GPU checkpoint",
            ),
            (
                "graphsage_weather_ddp_2gpu",
                run_root / "checkpoints" / "ddp_2gpu" / "best_rf_gnn_dynamic_sampler.pth",
                "2 GPU DDP checkpoint",
            ),
        ]
    rows = []
    existing = [(name, path, notes) for name, path, notes in checkpoint_specs if path.exists()]
    if not existing:
        return rows
    sid = int(__import__("numpy").load(data_dir / "sids.npy")[0]) if args.plot_station_id is None else args.plot_station_id
    plot_checkpoint = existing[-1]
    for model_name, checkpoint, notes in existing:
        checkpoint_dir = checkpoint.parent
        train_config = _checkpoint_train_config(checkpoint_dir)
        train_time_sec = train_config.get("total_time_sec", "")
        peak_gpu_mem_mb = train_config.get("peak_gpu_mem_mb", "")
        feature_ablation = train_config.get("feature_ablation", getattr(args, "feature_ablation", "none"))
        graph_mode = train_config.get("graph_mode", getattr(args, "graph_mode", "full"))
        feature_ablation_values = train_config.get("feature_ablation_values")
        model_type = train_config.get("model_type", getattr(args, "model_type", "graphsage"))
        if model_type == "mlp":
            model_name = model_name.replace("graphsage_weather", "mlp_no_graph", 1)
            ablation = "no_graph_mlp"
        else:
            ablation = ablation_label(feature_ablation, graph_mode)
            if ablation != "full" and model_name.startswith("graphsage_weather"):
                model_name = model_name.replace("graphsage_weather", f"graphsage_{ablation}", 1)
        make_plots = checkpoint == plot_checkpoint[1]
        rows.extend(
            evaluate_checkpoint_metrics(
                data_dir=data_dir,
                edge_index_path=data_dir / "step52_edge_index.pt",
                checkpoint_path=checkpoint,
                output_csv=None,
                run_id=run_root.name,
                max_eval_times=args.max_eval_times,
                plot_station_id=sid if make_plots else None,
                prediction_plot_path=run_root / "plots" / f"prediction_vs_truth_station_{sid}.png" if make_plots else None,
                prediction_csv_path=run_root / "predictions" / f"prediction_samples_station_{sid}.csv" if make_plots else None,
                error_heatmap_path=run_root / "plots" / "error_heatmap.png" if make_plots else None,
                model_name=model_name,
                ablation=ablation,
                feature_ablation=feature_ablation,
                graph_mode=graph_mode,
                feature_ablation_values=feature_ablation_values,
                model_type=model_type,
                normalization=train_config.get("normalization", getattr(args, "normalization", "none")),
                normalization_stats_path=train_config.get("normalization_stats_path", ""),
                temporal_encoding=train_config.get("temporal_encoding", {}).get(
                    "enabled",
                    not getattr(args, "no_temporal_encoding", False),
                )
                if isinstance(train_config.get("temporal_encoding"), dict)
                else bool(train_config.get("temporal_encoding", not getattr(args, "no_temporal_encoding", False))),
                station_embedding_dim=train_config.get("station_embedding_dim", getattr(args, "station_embedding_dim", 0)),
                hidden_dim=train_config.get("hidden_dim", getattr(args, "hidden_dim", 64)),
                num_layers=train_config.get("num_layers", getattr(args, "num_layers", 2)),
                dropout=train_config.get("dropout", getattr(args, "dropout", 0.3)),
                train_time_sec=train_time_sec,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                notes=f"{notes};model_type={model_type};ddp_shard_mode={train_config.get('ddp_shard_mode', 'unknown')}",
            )
        )
    write_metric_rows(run_root / "metrics" / "model_metrics.csv", rows)
    return rows


def run_logged(command, log_path, env=None, cwd=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def train_command(args, checkpoint_dir, metrics_csv, train_config_json):
    cmd = [
        args.python_bin,
        "-m",
        "model.train_sampler",
        "--data-dir",
        str(Path(args.run_root) / "pems_data"),
        "--edge-index-path",
        str(Path(args.run_root) / "pems_data" / "step52_edge_index.pt"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--metrics-csv",
        str(metrics_csv),
        "--train-config-json",
        str(train_config_json),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--time-stride",
        str(args.time_stride),
        "--train-fraction",
        str(args.train_fraction),
        "--val-fraction",
        str(args.val_fraction),
        "--val-period",
        str(args.val_period),
        "--max-train-times",
        str(args.max_train_times),
        "--max-val-times",
        str(args.max_val_times),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--accum-steps",
        str(args.accum_steps),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--ddp-shard-mode",
        args.ddp_shard_mode,
        "--feature-ablation",
        args.feature_ablation,
        "--graph-mode",
        args.graph_mode,
        "--normalization",
        args.normalization,
        "--station-embedding-dim",
        str(args.station_embedding_dim),
        "--model-type",
        args.model_type,
    ]
    if args.no_amp:
        cmd.append("--no-amp")
    if args.no_temporal_encoding:
        cmd.append("--no-temporal-encoding")
    return cmd


def run_single_training(args):
    checkpoint_dir = Path(args.run_root) / "checkpoints" / "single_gpu"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.single_gpu
    cmd = train_command(
        args,
        checkpoint_dir,
        checkpoint_dir / "epoch_metrics.csv",
        checkpoint_dir / "train_config.json",
    )
    return run_logged(cmd, Path(args.run_root) / "logs" / "single_gpu.log", env=env, cwd=args.project_root)


def run_ddp_training(args):
    checkpoint_dir = Path(args.run_root) / "checkpoints" / "ddp_2gpu"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.ddp_gpus
    cmd = [
        args.torchrun_bin,
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "model.train_sampler",
        "--data-dir",
        str(Path(args.run_root) / "pems_data"),
        "--edge-index-path",
        str(Path(args.run_root) / "pems_data" / "step52_edge_index.pt"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--metrics-csv",
        str(checkpoint_dir / "epoch_metrics.csv"),
        "--train-config-json",
        str(checkpoint_dir / "train_config.json"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--time-stride",
        str(args.time_stride),
        "--train-fraction",
        str(args.train_fraction),
        "--val-fraction",
        str(args.val_fraction),
        "--val-period",
        str(args.val_period),
        "--max-train-times",
        str(args.max_train_times),
        "--max-val-times",
        str(args.max_val_times),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--accum-steps",
        str(args.accum_steps),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--ddp-shard-mode",
        args.ddp_shard_mode,
        "--feature-ablation",
        args.feature_ablation,
        "--graph-mode",
        args.graph_mode,
        "--normalization",
        args.normalization,
        "--station-embedding-dim",
        str(args.station_embedding_dim),
        "--model-type",
        args.model_type,
    ]
    if args.no_amp:
        cmd.append("--no-amp")
    if args.no_temporal_encoding:
        cmd.append("--no-temporal-encoding")
    return run_logged(cmd, Path(args.run_root) / "logs" / "ddp_2gpu.log", env=env, cwd=args.project_root)


def run_pipeline(args):
    args.run_root = str(resolve_output_root(args.output_root, args.root_base))
    run_root = ensure_evidence_dirs(args.run_root)
    data_dir = run_root / "pems_data"
    if not args.skip_slice:
        create_data_slice(
            source_data_dir=args.source_data_dir,
            output_root=run_root,
            time_start=args.time_start,
            time_end_exclusive=args.time_end_exclusive,
            all_times=args.all_times,
            meta_csv=args.meta_csv,
        )
    write_train_config(run_root, args)
    write_run_metadata(run_root, args)
    write_baseline_metrics(
        data_dir,
        run_root / "metrics" / "baseline_metrics.csv",
        run_id=run_root.name,
        max_rf_train_samples=args.rf_train_samples,
        max_rf_test_samples=args.rf_test_samples,
        seed=args.seed,
        include_rf=not args.skip_rf,
    )
    if not args.skip_training:
        if args.skip_single:
            (run_root / "logs" / "single_gpu.log").write_text("single GPU training skipped by --skip-single\n")
        else:
            rc = run_single_training(args)
            if rc != 0:
                raise RuntimeError(f"single GPU training failed with exit code {rc}")
        if not args.skip_ddp:
            rc = run_ddp_training(args)
            if rc != 0:
                raise RuntimeError(f"DDP training failed with exit code {rc}")
        else:
            (run_root / "logs" / "ddp_2gpu.log").write_text("DDP training skipped by --skip-ddp\n")
    else:
        (run_root / "logs" / "single_gpu.log").write_text("single GPU training skipped by --skip-training\n")
        (run_root / "logs" / "ddp_2gpu.log").write_text("DDP training skipped by --skip-training\n")
    evaluate_available_checkpoints(args, data_dir, run_root)
    meta_csv = data_dir / "step01_d07_meta.csv"
    if meta_csv.exists():
        plot_topology(data_dir, meta_csv, run_root / "plots" / "topology.png")
    write_scaling_metrics(run_root)
    write_ablation_metrics(run_root)
    print(f"Evidence Pack written to {run_root}")
    return run_root


def add_common_args(parser):
    parser.add_argument("--source-data-dir", default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--root-base", default=DEFAULT_ROOT_BASE)
    parser.add_argument("--project-root", default="/scratch/lgong1/finalproject")
    parser.add_argument("--time-start", default=DEFAULT_TIME_START)
    parser.add_argument("--time-end-exclusive", default=DEFAULT_TIME_END_EXCLUSIVE)
    parser.add_argument("--all-times", action="store_true")
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-ref", default=None)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Create a complete Evidence Pack")
    add_common_args(run)
    run.add_argument("--python-bin", default=DEFAULT_PYTHON)
    run.add_argument("--torchrun-bin", default=DEFAULT_TORCHRUN)
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=1024)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--time-stride", type=int, default=1)
    run.add_argument("--train-fraction", type=float, default=0.70)
    run.add_argument("--val-fraction", type=float, default=0.15)
    run.add_argument("--val-period", type=int, default=1)
    run.add_argument("--max-train-times", type=int, default=512)
    run.add_argument("--max-val-times", type=int, default=128)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--hidden-dim", type=int, default=64)
    run.add_argument("--num-layers", type=int, default=2)
    run.add_argument("--dropout", type=float, default=0.3)
    run.add_argument("--accum-steps", type=int, default=1)
    run.add_argument("--early-stop-patience", type=int, default=5)
    run.add_argument("--no-amp", action="store_true")
    run.add_argument(
        "--ddp-shard-mode",
        choices=("flat-batch", "legacy-time-node"),
        default="flat-batch",
    )
    run.add_argument("--feature-ablation", choices=("none", "without_weather"), default="none")
    run.add_argument("--graph-mode", choices=("full", "self_loop"), default="full")
    run.add_argument("--normalization", choices=("none", "station-wise-flow"), default="station-wise-flow")
    run.add_argument("--no-temporal-encoding", action="store_true")
    run.add_argument("--station-embedding-dim", type=int, default=16)
    run.add_argument("--model-type", choices=("graphsage", "mlp"), default="graphsage")
    run.add_argument("--single-gpu", default="2")
    run.add_argument("--ddp-gpus", default="2,3")
    run.add_argument("--skip-slice", action="store_true")
    run.add_argument("--skip-training", action="store_true")
    run.add_argument("--skip-single", action="store_true")
    run.add_argument("--skip-ddp", action="store_true")
    run.add_argument("--skip-rf", action="store_true")
    run.add_argument("--rf-train-samples", type=int, default=100000)
    run.add_argument("--rf-test-samples", type=int, default=100000)
    run.add_argument("--max-eval-times", type=int, default=0)
    run.add_argument("--checkpoint-path", default=None)
    run.add_argument("--plot-station-id", type=int, default=None)
    run.set_defaults(func=run_pipeline)

    prep = sub.add_parser("prepare-slice", help="Create only the reproducible data slice")
    add_common_args(prep)
    prep.set_defaults(
        func=lambda args: create_data_slice(
            source_data_dir=args.source_data_dir,
            output_root=resolve_output_root(args.output_root, args.root_base),
            time_start=args.time_start,
            time_end_exclusive=args.time_end_exclusive,
            all_times=args.all_times,
            meta_csv=args.meta_csv,
        )
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
