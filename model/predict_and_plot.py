#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict 5/15/30 minute flow for a station or internal graph node and plot it.
"""

import argparse
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import NeighborSampler

from .dataset_full import DATA_DIR, RFGraphDatasetFull
from .gnn_final import MultiHeadRFGraphSAGEDyn

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
SIZES = {
    "5min": [8, 8],
    "15min": [12, 12],
    "30min": [16, 16],
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
        "--checkpoint-path",
        "--model-path",
        dest="checkpoint_path",
        default="best_rf_gnn_dynamic_sampler.pth",
        help="Model checkpoint path",
    )
    parser.add_argument("--output-path", default=None, help="PNG output path")
    parser.add_argument("--station-id", type=int, default=None, help="Real PeMS station ID")
    parser.add_argument("--node-idx", type=int, default=None, help="Internal graph node index")
    parser.add_argument("--time-idx", type=int, default=None, help="Time step index")
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help='Timestamp from timestamps.npy, e.g. "2024-01-01 00:05:00"',
    )
    parser.add_argument("--flow", type=float, default=None, help="Optional current flow override")
    parser.add_argument("--occupancy", "--occ", dest="occupancy", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None, help="Optional current speed override")
    parser.add_argument("-k", type=int, default=5, help="Number of graph neighbors to include")
    return parser.parse_args(argv)


def resolve_prediction_node(ds, station_id=None, node_idx=None):
    if station_id is None and node_idx is None:
        raise ValueError("Specify exactly one of --station-id or --node-idx")
    if station_id is not None and node_idx is not None:
        raise ValueError("Specify only one of --station-id or --node-idx")
    if node_idx is not None:
        node_idx = int(node_idx)
        if node_idx < 0 or node_idx >= ds.N:
            raise IndexError(f"Node index {node_idx} is outside [0, {ds.N})")
        return node_idx
    return ds.station_id_to_node_idx(station_id)


def apply_realtime_overrides(features, node_idx, flow=None, occupancy=None, speed=None):
    # X_ext feature order: flow, occupancy, speed, tavg, pcpn, is_weekend.
    if flow is not None:
        features[node_idx, 0] = float(flow)
    if occupancy is not None:
        features[node_idx, 1] = float(occupancy)
    if speed is not None:
        features[node_idx, 2] = float(speed)
    return features


def resolve_time_idx(ds, time_idx=None, timestamp=None):
    if time_idx is not None and timestamp is not None:
        raise ValueError("Specify only one of --time-idx or --time")
    if time_idx is not None:
        time_idx = int(time_idx)
        if time_idx < 0 or time_idx >= ds.T:
            raise IndexError(f"Time index {time_idx} is outside [0, {ds.T})")
        return time_idx
    if timestamp is None:
        raise ValueError("Specify one of --time-idx or --time")
    return ds.time_to_idx(timestamp)


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    cleaned = OrderedDict()
    for key, value in state.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        cleaned[name] = value
    model.load_state_dict(cleaned)


def main(argv=None):
    args = parse_args(argv)
    args.edge_index_path = args.edge_index_path or os.path.join(args.data_dir, "step52_edge_index.pt")

    ds = RFGraphDatasetFull(data_dir=args.data_dir)
    time_idx = resolve_time_idx(ds, time_idx=args.time_idx, timestamp=args.time)
    target_node_idx = resolve_prediction_node(ds, args.station_id, args.node_idx)
    target_station_id = ds.node_idx_to_station_id(target_node_idx)

    features = ds.X[time_idx].clone().float()
    apply_realtime_overrides(
        features,
        target_node_idx,
        flow=args.flow,
        occupancy=args.occupancy,
        speed=args.speed,
    )

    edge_index = torch.load(args.edge_index_path, map_location="cpu").long().contiguous()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiHeadRFGraphSAGEDyn(
        in_dim=features.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    row, col = edge_index
    neighbor_nodes = col[row == target_node_idx].unique().tolist()[: args.k]
    all_node_ids = [target_node_idx] + [int(node) for node in neighbor_nodes]
    station_ids = [ds.node_idx_to_station_id(node) for node in all_node_ids]

    preds = {}
    for window, sizes in SIZES.items():
        sampler = NeighborSampler(
            edge_index,
            sizes=sizes,
            node_idx=torch.tensor(all_node_ids, dtype=torch.long),
            batch_size=len(all_node_ids),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        batch_size, n_id, adjs = next(iter(sampler))
        x = features[n_id].to(device)
        adjs = [adj.to(device) for adj in adjs]
        with torch.no_grad():
            out = model(x, adjs, head=window)
        preds[window] = out[:batch_size].squeeze(1).cpu().tolist()

    import pandas as pd

    df = pd.DataFrame(
        {
            "station_id": station_ids,
            "node_idx": all_node_ids,
            "flow_5min": preds["5min"],
            "flow_15min": preds["15min"],
            "flow_30min": preds["30min"],
        }
    ).set_index("station_id")

    windows = [5, 15, 30]
    plt.figure(figsize=(6, 4))
    for station_id in df.index:
        y = [df.loc[station_id, f"flow_{window}min"] for window in windows]
        label = "Target" if int(station_id) == target_station_id else f"Neighbor {station_id}"
        plt.plot(windows, y, marker="o", label=label)
    plt.xlabel("Prediction window (minutes)")
    plt.ylabel("Flow prediction")
    plt.title(f"Station {target_station_id} + {len(neighbor_nodes)} neighbors")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    output_path = args.output_path or f"flow_pred_{target_station_id}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    print(df)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
