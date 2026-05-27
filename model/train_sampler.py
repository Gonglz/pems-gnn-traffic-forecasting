#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal NeighborSampler training for the multi-head RF GraphSAGE model.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

from .dataset_full import DATA_DIR, DELTA15, DELTA30, DELTA5, RFGraphDatasetFull
from .gnn_final import MultiHeadRFGraphSAGEDyn, MultiHeadRFMLP
from .p2_quality import (
    MODEL_TYPE_CHOICES,
    NORMALIZATION_CHOICES,
    NORMALIZATION_NONE,
    NORMALIZATION_STATION_WISE_FLOW,
    apply_input_flow_normalization_,
    apply_temporal_encoding_,
    audit_forecast_task,
    compute_stationwise_flow_stats,
    normalization_summary,
    normalize_target,
    save_normalization_stats,
)
from .training_modes import (
    FEATURE_ABLATION_CHOICES,
    GRAPH_MODE_CHOICES,
    ablation_label,
    apply_feature_ablation_,
    apply_graph_mode,
)

EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 1024
ACCUM_STEPS = 1
NUM_WORKERS = 4
USE_AMP = True
VAL_PERIOD = 2
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 5
SIZES = {
    "5min": [8, 8],
    "15min": [12, 12],
    "30min": [16, 16],
}
HEAD_DELTAS = {
    "5min": DELTA5,
    "15min": DELTA15,
    "30min": DELTA30,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing X_ext/Y/sids/neighbors")
    parser.add_argument(
        "--edge-index-path",
        default=None,
        help="Path to topo edge_index .pt. Defaults to <data-dir>/step52_edge_index.pt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint-output-dir",
        dest="checkpoint_dir",
        default=".",
        help="Directory for best/last checkpoints and loss curve",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--max-train-times",
        type=int,
        default=None,
        help="Limit training time indices after split/stride. Use 0 for all.",
    )
    parser.add_argument(
        "--max-val-times",
        type=int,
        default=None,
        help="Limit validation time indices after split/stride. Use 0 for all.",
    )
    parser.add_argument("--time-stride", type=int, default=1, help="Use every Nth eligible time index")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--val-period", type=int, default=VAL_PERIOD)
    parser.add_argument("--lr-patience", type=int, default=LR_PATIENCE)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--ddp-shard-mode",
        choices=("flat-batch", "legacy-time-node"),
        default="flat-batch",
        help="DDP work partitioning. flat-batch shards (time_idx, seed-node-batch) fairly.",
    )
    parser.add_argument(
        "--feature-ablation",
        choices=FEATURE_ABLATION_CHOICES,
        default="none",
        help="Feature ablation applied consistently to train/eval.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=GRAPH_MODE_CHOICES,
        default="full",
        help="Graph structure for training. self_loop removes cross-station neighbors.",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default=NORMALIZATION_STATION_WISE_FLOW,
        help="P2 normalization mode. station-wise-flow uses train-only flow stats.",
    )
    parser.add_argument(
        "--no-temporal-encoding",
        action="store_true",
        help="Disable appended sin/cos temporal features.",
    )
    parser.add_argument("--station-embedding-dim", type=int, default=16)
    parser.add_argument("--model-type", choices=MODEL_TYPE_CHOICES, default="graphsage")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Rank-0 CSV path for per-epoch train/val MSE and timing.",
    )
    parser.add_argument(
        "--train-config-json",
        default=None,
        help="Rank-0 JSON path for resolved training configuration and summary.",
    )
    return parser.parse_args(argv)


def _normalize_limit(limit):
    if limit is None or limit <= 0:
        return None
    return limit


def _limit_indices(indices, limit):
    limit = _normalize_limit(limit)
    if limit is None:
        return list(indices)
    return list(indices)[:limit]


def build_time_splits(
    total_times,
    max_horizon=max(HEAD_DELTAS.values()),
    time_stride=1,
    max_train_times=None,
    max_val_times=None,
    train_fraction=0.8,
    val_fraction=None,
):
    if total_times <= max_horizon:
        raise ValueError(
            f"Need more than {max_horizon} time steps, got total_times={total_times}"
        )
    stride = max(1, int(time_stride))
    valid_times = list(range(0, total_times - max_horizon, stride))
    if not valid_times:
        raise ValueError("No valid time indices after applying horizon and stride")

    split = int(train_fraction * len(valid_times))
    if len(valid_times) > 1:
        split = min(max(split, 1), len(valid_times) - 1)
    train_times = valid_times[:split]
    if val_fraction is None:
        val_times = valid_times[split:]
    else:
        val_end = int((float(train_fraction) + float(val_fraction)) * len(valid_times))
        if len(valid_times) > 2:
            val_end = min(max(val_end, split + 1), len(valid_times) - 1)
        else:
            val_end = len(valid_times)
        val_times = valid_times[split:val_end]

    return (
        _limit_indices(train_times, max_train_times),
        _limit_indices(val_times, max_val_times),
    )


def partition_indices_for_rank(indices, rank=0, world_size=1):
    indices = list(indices)
    if world_size <= 1:
        return indices
    return indices[rank::world_size]


def rank_seed_nodes(num_nodes, rank=0, world_size=1):
    nodes = torch.arange(num_nodes, dtype=torch.long)
    if world_size <= 1:
        return nodes
    return nodes[rank::world_size]


def count_node_batches(num_nodes, batch_size):
    batch_size = max(1, int(batch_size))
    return (int(num_nodes) + batch_size - 1) // batch_size


def flat_batch_work_items(times, num_node_batches, rank=0, world_size=1):
    """Shard flattened (time_idx, seed_batch_idx) work without rank overlap."""
    times = list(map(int, times))
    work = [
        (time_idx, batch_idx)
        for time_idx in times
        for batch_idx in range(int(num_node_batches))
    ]
    if world_size <= 1:
        return work
    return work[int(rank) :: int(world_size)]


def count_pairs_for_work_items(work_items, num_nodes, batch_size):
    total = 0
    batch_size = max(1, int(batch_size))
    for _, batch_idx in work_items:
        start = int(batch_idx) * batch_size
        total += max(0, min(batch_size, int(num_nodes) - start))
    return total


def setup_distributed():
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not use_ddp:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, False, None

    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, dist.get_rank(), dist.get_world_size(), True, local_rank


def cleanup_distributed(use_ddp):
    if use_ddp:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def reduce_sum(value, device, use_ddp):
    if not use_ddp:
        return float(value)
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def make_sampler(edge_index, node_ids, sizes, batch_size, shuffle, num_workers):
    return NeighborSampler(
        edge_index,
        sizes=sizes,
        node_idx=torch.as_tensor(node_ids, dtype=torch.long),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def batch_xy(x_full, y_full, time_idx, node_ids, batch_size, delta, device):
    seed_nodes = node_ids[:batch_size]
    x = x_full[time_idx, node_ids].to(device, non_blocking=True)
    y = y_full[time_idx + delta, seed_nodes].unsqueeze(-1).to(device, non_blocking=True)
    return x, y, seed_nodes


def node_batch_ids(num_nodes, batch_idx, batch_size):
    start = int(batch_idx) * int(batch_size)
    end = min(int(num_nodes), start + int(batch_size))
    return torch.arange(start, end, dtype=torch.long)


def build_model(args, ds):
    model_kwargs = {
        "in_dim": ds.F,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_nodes": ds.N,
        "station_embedding_dim": args.station_embedding_dim,
    }
    if args.model_type == "mlp":
        return MultiHeadRFMLP(**model_kwargs)
    return MultiHeadRFGraphSAGEDyn(**model_kwargs)


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(unwrapped.state_dict(), path)


def write_epoch_metrics(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "epoch",
        "phase",
        "mse",
        "steps",
        "pairs",
        "pairs_per_sec",
        "phase_time_sec",
        "elapsed_sec",
        "world_size",
        "peak_gpu_mem_mb",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_train_config(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def peak_gpu_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def run_mlp_epoch(
    model,
    ds,
    times,
    args,
    device,
    train,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    rank=0,
    seed_node_ids=None,
    work_items=None,
    normalization_stats=None,
):
    if seed_node_ids is None:
        seed_node_ids = torch.arange(ds.N, dtype=torch.long)
    if train:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_steps = 0
    total_pairs = 0
    accum_counter = 0

    if work_items is None:
        local_batches = count_node_batches(len(seed_node_ids), args.batch_size)
        iterator = [
            (int(time_idx), int(batch_idx))
            for time_idx in times
            for batch_idx in range(local_batches)
        ]
        batch_nodes = lambda batch_idx: seed_node_ids[
            int(batch_idx) * args.batch_size : min(len(seed_node_ids), (int(batch_idx) + 1) * args.batch_size)
        ]
    else:
        iterator = work_items
        batch_nodes = lambda batch_idx: node_batch_ids(ds.N, batch_idx, args.batch_size)

    if rank == 0 and (sys.stdout.isatty() or sys.stderr.isatty()):
        iterator = tqdm(iterator, desc="mlp-batches", ncols=100, leave=False)

    for time_idx, batch_idx in iterator:
        seed_nodes = batch_nodes(batch_idx)
        if len(seed_nodes) == 0:
            continue
        node_ids_device = seed_nodes.to(device, non_blocking=True)
        x = ds.X[int(time_idx), seed_nodes].to(device, non_blocking=True)
        head_batches = {}
        targets = {}
        for head, delta in HEAD_DELTAS.items():
            y = ds.Y[int(time_idx) + delta, seed_nodes].unsqueeze(-1).to(device, non_blocking=True)
            head_batches[head] = (x, node_ids_device)
            targets[head] = (y, seed_nodes)

        context = autocast(enabled=amp_enabled) if train else torch.no_grad()
        with context:
            preds = model(head_batches=head_batches)
            raw_losses = [
                F.mse_loss(
                    preds[head],
                    normalize_target(
                        targets[head][0],
                        normalization_stats,
                        head,
                        targets[head][1],
                        device,
                    ),
                )
                for head in SIZES.keys()
            ]

        if train:
            step_loss = sum(raw_losses) / len(raw_losses)
            scaled_loss = step_loss / args.accum_steps
            scaler.scale(scaled_loss).backward()
            accum_counter += 1
            if accum_counter % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            step_loss = sum(loss.detach() for loss in raw_losses) / len(raw_losses)

        total_loss += float(step_loss.detach().item())
        total_steps += 1
        total_pairs += int(len(seed_nodes))

    if train and accum_counter % args.accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss, total_steps, total_pairs


def run_epoch(
    model,
    ds,
    edge_index,
    times,
    args,
    device,
    train,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    rank=0,
    seed_node_ids=None,
    work_items=None,
    normalization_stats=None,
):
    model.train(train)
    forward_model = model
    if not train and isinstance(model, DistributedDataParallel):
        # Evaluation has no gradients to synchronize. Using the wrapped DDP
        # module here can hang when a rank has no held-out time slice.
        forward_model = model.module
    if seed_node_ids is None:
        seed_node_ids = torch.arange(ds.N, dtype=torch.long)
    if args.model_type == "mlp":
        return run_mlp_epoch(
            forward_model,
            ds,
            times,
            args,
            device,
            train,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            rank=rank,
            seed_node_ids=seed_node_ids,
            work_items=work_items,
            normalization_stats=normalization_stats,
        )
    samplers = {
        head: make_sampler(
            edge_index=edge_index,
            node_ids=seed_node_ids,
            sizes=sizes,
            batch_size=args.batch_size,
            shuffle=train and work_items is None,
            num_workers=args.num_workers,
        )
        for head, sizes in SIZES.items()
    }
    cached_batches = None
    if work_items is not None:
        cached_batches = list(zip(samplers["5min"], samplers["15min"], samplers["30min"]))

    if train:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_steps = 0
    total_pairs = 0
    accum_counter = 0
    iterator = work_items if work_items is not None else times
    if rank == 0 and (sys.stdout.isatty() or sys.stderr.isatty()):
        desc = "flat-batches" if work_items is not None else "times"
        iterator = tqdm(iterator, desc=desc, ncols=100, leave=False)

    for item in iterator:
        if work_items is None:
            time_idx = int(item)
            batch_iter = zip(samplers["5min"], samplers["15min"], samplers["30min"])
        else:
            time_idx, batch_idx = item
            batch_iter = [cached_batches[int(batch_idx)]]
        for batches in batch_iter:
            step_loss = 0.0
            head_batches = {}
            targets = {}
            seed_count = 0
            for head, (batch_size, node_ids, adjs) in zip(SIZES.keys(), batches):
                if not seed_count:
                    seed_count = int(batch_size)
                x, y, seed_nodes = batch_xy(
                    ds.X,
                    ds.Y,
                    time_idx,
                    node_ids,
                    batch_size,
                    HEAD_DELTAS[head],
                    device,
                )
                adjs = [adj.to(device) for adj in adjs]
                head_batches[head] = (x, adjs, node_ids.to(device, non_blocking=True))
                targets[head] = (y, seed_nodes)

            context = autocast(enabled=amp_enabled) if train else torch.no_grad()
            with context:
                preds = forward_model(head_batches=head_batches)
                raw_losses = [
                    F.mse_loss(
                        preds[head],
                        normalize_target(
                            targets[head][0],
                            normalization_stats,
                            head,
                            targets[head][1],
                            device,
                        ),
                    )
                    for head in SIZES.keys()
                ]

            if train:
                step_loss = sum(raw_losses) / len(raw_losses)
                scaled_loss = step_loss / args.accum_steps
                scaler.scale(scaled_loss).backward()
                accum_counter += 1
                if accum_counter % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                step_loss = sum(loss.detach() for loss in raw_losses) / len(raw_losses)

            total_loss += float(step_loss.detach().item())
            total_steps += 1
            total_pairs += seed_count

    if train and accum_counter % args.accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss, total_steps, total_pairs


def train(argv=None):
    args = parse_args(argv)
    args.edge_index_path = args.edge_index_path or os.path.join(args.data_dir, "step52_edge_index.pt")
    args.accum_steps = max(1, int(args.accum_steps))
    args.metrics_csv = args.metrics_csv or os.path.join(args.checkpoint_dir, "epoch_metrics.csv")
    args.train_config_json = args.train_config_json or os.path.join(args.checkpoint_dir, "train_config.json")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, rank, world_size, use_ddp, local_rank = setup_distributed()
    try:
        if rank == 0:
            print(f"[Device] {device} rank={rank}/{world_size}")
        start_time = time.time()

        ds = RFGraphDatasetFull(data_dir=args.data_dir)
        train_times, val_times = build_time_splits(
            total_times=ds.T,
            time_stride=args.time_stride,
            max_train_times=args.max_train_times,
            max_val_times=args.max_val_times,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        if not train_times:
            raise ValueError("No training time indices selected")
        if not val_times:
            raise ValueError("No validation time indices selected")

        task_audit = audit_forecast_task(ds, train_times=train_times)
        feature_ablation_metadata = apply_feature_ablation_(
            ds.X,
            train_times,
            args.feature_ablation,
        )
        temporal_metadata = apply_temporal_encoding_(
            ds,
            enabled=not args.no_temporal_encoding,
        )
        normalization_stats = None
        normalization_stats_path = ""
        normalization_metadata = {"mode": NORMALIZATION_NONE}
        if args.normalization == NORMALIZATION_STATION_WISE_FLOW:
            normalization_stats = compute_stationwise_flow_stats(ds.X, ds.Y, train_times, HEAD_DELTAS)
            normalization_stats_path = os.path.join(args.checkpoint_dir, "normalization_stats.npz")
            if rank == 0:
                save_normalization_stats(normalization_stats_path, normalization_stats)
            apply_input_flow_normalization_(ds.X, normalization_stats)
            normalization_metadata = normalization_summary(normalization_stats, normalization_stats_path)

        if args.model_type == "mlp":
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.load(args.edge_index_path, map_location="cpu").long().contiguous()
            edge_index = apply_graph_mode(edge_index, ds.N, args.graph_mode)

        num_node_batches = count_node_batches(ds.N, args.batch_size)
        if args.ddp_shard_mode == "flat-batch":
            seed_node_ids = torch.arange(ds.N, dtype=torch.long)
            rank_train_work = flat_batch_work_items(train_times, num_node_batches, rank, world_size)
            rank_val_work = flat_batch_work_items(val_times, num_node_batches, rank, world_size)
            rank_train_times = sorted({time_idx for time_idx, _ in rank_train_work})
            rank_val_times = sorted({time_idx for time_idx, _ in rank_val_work})
            rank_train_pairs = count_pairs_for_work_items(rank_train_work, ds.N, args.batch_size)
            rank_val_pairs = count_pairs_for_work_items(rank_val_work, ds.N, args.batch_size)
            rank_train_steps = len(rank_train_work)
            rank_val_steps = len(rank_val_work)
        else:
            rank_train_work = None
            rank_val_work = None
            rank_train_times = partition_indices_for_rank(train_times, rank, world_size)
            rank_val_times = partition_indices_for_rank(val_times, rank, world_size)
            seed_node_ids = rank_seed_nodes(ds.N, rank, world_size)
            if len(seed_node_ids) == 0:
                raise ValueError(f"Rank {rank} has no seed nodes for N={ds.N}, world_size={world_size}")
            rank_node_batches = count_node_batches(len(seed_node_ids), args.batch_size)
            rank_train_pairs = len(rank_train_times) * int(len(seed_node_ids))
            rank_val_pairs = len(rank_val_times) * int(len(seed_node_ids))
            rank_train_steps = len(rank_train_times) * rank_node_batches
            rank_val_steps = len(rank_val_times) * rank_node_batches

        global_train_pairs = len(train_times) * ds.N
        global_val_pairs = len(val_times) * ds.N
        global_train_steps = len(train_times) * num_node_batches
        global_val_steps = len(val_times) * num_node_batches
        rank_work_summary = {
            "rank": int(rank),
            "train_times": int(len(rank_train_times)),
            "val_times": int(len(rank_val_times)),
            "train_pairs": int(rank_train_pairs),
            "val_pairs": int(rank_val_pairs),
            "train_steps": int(rank_train_steps),
            "val_steps": int(rank_val_steps),
        }
        rank_work_by_rank = [rank_work_summary]
        if use_ddp:
            import torch.distributed as dist

            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, rank_work_summary)
            rank_work_by_rank = gathered

        if rank == 0:
            print(
                f"Dataset: T={ds.T}, N={ds.N}, F={ds.F}, "
                f"train_times={len(train_times)}, val_times={len(val_times)}, "
                f"time_stride={args.time_stride}, world_size={world_size}, "
                f"ddp_shard_mode={args.ddp_shard_mode}, "
                f"feature_ablation={args.feature_ablation}, graph_mode={args.graph_mode}, "
                f"model_type={args.model_type}, normalization={args.normalization}, "
                f"station_embedding_dim={args.station_embedding_dim}"
            )

        model = build_model(args, ds).to(device)
        if use_ddp:
            kwargs = {"find_unused_parameters": False}
            if device.type == "cuda":
                kwargs["device_ids"] = [local_rank]
            model = DistributedDataParallel(model, **kwargs)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=args.lr_patience,
            verbose=(rank == 0),
        )
        amp_enabled = USE_AMP and not args.no_amp and device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats(device)

        train_losses = []
        val_losses = []
        metric_rows = []
        best_val = float("inf")
        epochs_no_improve = 0

        if rank == 0:
            write_train_config(
                args.train_config_json,
                {
                    "data_dir": args.data_dir,
                    "edge_index_path": args.edge_index_path,
                    "checkpoint_dir": args.checkpoint_dir,
                    "T": ds.T,
                    "N": ds.N,
                    "F": ds.F,
                    "train_times": len(train_times),
                    "val_times": len(val_times),
                    "rank_train_times": len(rank_train_times),
                    "rank_val_times": len(rank_val_times),
                    "world_size": world_size,
                    "seed_nodes_per_rank": int(len(seed_node_ids)),
                    "num_node_batches": int(num_node_batches),
                    "global_time_node_pairs": {
                        "train": int(global_train_pairs),
                        "val": int(global_val_pairs),
                    },
                    "rank_pairs": rank_work_by_rank,
                    "steps_per_epoch": {
                        "train": int(global_train_steps),
                        "val": int(global_val_steps),
                        "rank_train": int(rank_train_steps),
                        "rank_val": int(rank_val_steps),
                    },
                    "ddp_shard_mode": args.ddp_shard_mode,
                    "feature_ablation": args.feature_ablation,
                    "feature_ablation_values": feature_ablation_metadata,
                    "graph_mode": args.graph_mode,
                    "edge_count": int(edge_index.shape[1]),
                    "ablation": "no_graph_mlp" if args.model_type == "mlp" else ablation_label(args.feature_ablation, args.graph_mode),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "time_stride": args.time_stride,
                    "max_train_times": args.max_train_times,
                    "max_val_times": args.max_val_times,
                    "train_fraction": args.train_fraction,
                    "val_fraction": args.val_fraction,
                    "lr": args.lr,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "accum_steps": args.accum_steps,
                    "val_period": args.val_period,
                    "amp_enabled": bool(amp_enabled),
                    "model_type": args.model_type,
                    "normalization": args.normalization,
                    "normalization_stats_path": normalization_stats_path,
                    "normalization_summary": normalization_metadata,
                    "temporal_encoding": temporal_metadata,
                    "feature_order": list(getattr(ds, "feature_order", [])),
                    "station_embedding_dim": args.station_embedding_dim,
                    "task_audit": task_audit,
                    "sampler_sizes": SIZES,
                    "head_deltas": HEAD_DELTAS,
                    "status": "started",
                },
            )

        for epoch in range(1, args.epochs + 1):
            if rank == 0:
                print(f"\n=== Epoch {epoch}/{args.epochs} ===")

            join_context = nullcontext()
            if use_ddp:
                from torch.distributed.algorithms.join import Join

                join_context = Join([model])
            with join_context:
                phase_start = time.time()
                local_train_loss, local_train_steps, local_train_pairs = run_epoch(
                    model,
                    ds,
                    edge_index,
                    rank_train_times,
                    args,
                    device,
                    train=True,
                    optimizer=optimizer,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    rank=rank,
                    seed_node_ids=seed_node_ids,
                    work_items=rank_train_work,
                    normalization_stats=normalization_stats,
                )
                local_train_time = time.time() - phase_start

            train_loss_sum = reduce_sum(local_train_loss, device, use_ddp)
            train_step_sum = reduce_sum(local_train_steps, device, use_ddp)
            train_pair_sum = reduce_sum(local_train_pairs, device, use_ddp)
            train_time_sum = reduce_sum(local_train_time, device, use_ddp)
            avg_train_loss = train_loss_sum / max(train_step_sum, 1.0)
            train_losses.append(avg_train_loss)
            if rank == 0:
                print(f"[Epoch {epoch}] train MSE = {avg_train_loss:.6f}")
                metric_rows.append(
                    {
                        "epoch": epoch,
                        "phase": "train",
                        "mse": avg_train_loss,
                        "steps": int(train_step_sum),
                        "pairs": int(train_pair_sum),
                        "pairs_per_sec": float(train_pair_sum) / max(train_time_sum / max(world_size, 1), 1e-9),
                        "phase_time_sec": train_time_sum / max(world_size, 1),
                        "elapsed_sec": time.time() - start_time,
                        "world_size": world_size,
                        "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
                    }
                )
                write_epoch_metrics(args.metrics_csv, metric_rows)

            if epoch % args.val_period == 0:
                phase_start = time.time()
                local_val_loss, local_val_steps, local_val_pairs = run_epoch(
                    model,
                    ds,
                    edge_index,
                    rank_val_times,
                    args,
                    device,
                    train=False,
                    rank=rank,
                    seed_node_ids=seed_node_ids,
                    work_items=rank_val_work,
                    normalization_stats=normalization_stats,
                )
                local_val_time = time.time() - phase_start
                val_loss_sum = reduce_sum(local_val_loss, device, use_ddp)
                val_step_sum = reduce_sum(local_val_steps, device, use_ddp)
                val_pair_sum = reduce_sum(local_val_pairs, device, use_ddp)
                val_time_sum = reduce_sum(local_val_time, device, use_ddp)
                avg_val_loss = val_loss_sum / max(val_step_sum, 1.0)
                val_losses.append(avg_val_loss)
                if rank == 0:
                    print(f"[Epoch {epoch}] val MSE   = {avg_val_loss:.6f}")
                    metric_rows.append(
                        {
                            "epoch": epoch,
                            "phase": "val",
                            "mse": avg_val_loss,
                            "steps": int(val_step_sum),
                            "pairs": int(val_pair_sum),
                            "pairs_per_sec": float(val_pair_sum) / max(val_time_sum / max(world_size, 1), 1e-9),
                            "phase_time_sec": val_time_sum / max(world_size, 1),
                            "elapsed_sec": time.time() - start_time,
                            "world_size": world_size,
                            "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
                        }
                    )
                    write_epoch_metrics(args.metrics_csv, metric_rows)

                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val:
                    best_val = avg_val_loss
                    epochs_no_improve = 0
                    if rank == 0:
                        best_path = os.path.join(
                            args.checkpoint_dir, "best_rf_gnn_dynamic_sampler.pth"
                        )
                        save_checkpoint(model, best_path)
                        print(f"  (saved best model to {best_path})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.early_stop_patience:
                        if rank == 0:
                            print(f"Early stopping after {epoch} epochs.")
                        break

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            elapsed = time.time() - start_time
            print(f"\nTotal training time: {elapsed / 60:.2f} minutes")

            epochs = list(range(1, len(train_losses) + 1))
            plt.figure()
            plt.plot(epochs, train_losses, label="train")
            val_epochs = list(range(args.val_period, args.val_period * len(val_losses) + 1, args.val_period))
            if val_losses:
                plt.plot(val_epochs, val_losses, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.title("Training and Validation Loss")
            plt.grid(True)
            loss_path = os.path.join(args.checkpoint_dir, "loss_curve.png")
            plt.savefig(loss_path)
            print(f"Saved {loss_path}")

            last_path = os.path.join(args.checkpoint_dir, "rf_gnn_dynamic_sampler_last.pth")
            save_checkpoint(model, last_path)
            print(f"Saved checkpoint {last_path}")
            write_train_config(
                args.train_config_json,
                {
                    "data_dir": args.data_dir,
                    "edge_index_path": args.edge_index_path,
                    "checkpoint_dir": args.checkpoint_dir,
                    "T": ds.T,
                    "N": ds.N,
                    "F": ds.F,
                    "train_times": len(train_times),
                    "val_times": len(val_times),
                    "rank_train_times": len(rank_train_times),
                    "rank_val_times": len(rank_val_times),
                    "world_size": world_size,
                    "seed_nodes_per_rank": int(len(seed_node_ids)),
                    "num_node_batches": int(num_node_batches),
                    "global_time_node_pairs": {
                        "train": int(global_train_pairs),
                        "val": int(global_val_pairs),
                    },
                    "rank_pairs": rank_work_by_rank,
                    "steps_per_epoch": {
                        "train": int(global_train_steps),
                        "val": int(global_val_steps),
                        "rank_train": int(rank_train_steps),
                        "rank_val": int(rank_val_steps),
                    },
                    "ddp_shard_mode": args.ddp_shard_mode,
                    "feature_ablation": args.feature_ablation,
                    "feature_ablation_values": feature_ablation_metadata,
                    "graph_mode": args.graph_mode,
                    "edge_count": int(edge_index.shape[1]),
                    "ablation": "no_graph_mlp" if args.model_type == "mlp" else ablation_label(args.feature_ablation, args.graph_mode),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "time_stride": args.time_stride,
                    "max_train_times": args.max_train_times,
                    "max_val_times": args.max_val_times,
                    "train_fraction": args.train_fraction,
                    "val_fraction": args.val_fraction,
                    "lr": args.lr,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "accum_steps": args.accum_steps,
                    "val_period": args.val_period,
                    "amp_enabled": bool(amp_enabled),
                    "model_type": args.model_type,
                    "normalization": args.normalization,
                    "normalization_stats_path": normalization_stats_path,
                    "normalization_summary": normalization_metadata,
                    "temporal_encoding": temporal_metadata,
                    "feature_order": list(getattr(ds, "feature_order", [])),
                    "station_embedding_dim": args.station_embedding_dim,
                    "task_audit": task_audit,
                    "sampler_sizes": SIZES,
                    "head_deltas": HEAD_DELTAS,
                    "best_val_mse": best_val,
                    "total_time_sec": elapsed,
                    "peak_gpu_mem_mb": peak_gpu_memory_mb(device),
                    "metrics_csv": args.metrics_csv,
                    "best_checkpoint": os.path.join(args.checkpoint_dir, "best_rf_gnn_dynamic_sampler.pth"),
                    "last_checkpoint": last_path,
                    "status": "completed",
                },
            )
    finally:
        cleanup_distributed(use_ddp)


if __name__ == "__main__":
    train()
