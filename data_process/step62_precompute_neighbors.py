#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute weighted graph neighborhoods and static multi-scale edge_index files.

All output writes happen inside main(). Defaults preserve the production
/scratch paths; CLI arguments or environment variables can redirect paths for
small local samples.
"""

import argparse
import math
import os
import pickle
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
import torch


DEFAULT_BASE = "/scratch/lgong1/finalproject/pems_data"
DELTAS = {"5min": 1, "15min": 3, "30min": 6}
HOURS = {name: delta * 5 / 60 for name, delta in DELTAS.items()}
DEFAULT_LENGTH = 1.0


def _env(name, default=None):
    return os.environ.get(name, default)


def _resolve_path(value, base_dir, filename, env_name):
    return value or _env(env_name) or os.path.join(base_dir, filename)


def _normalise_meta(meta):
    out = meta.copy()
    rename = {}
    for col in out.columns:
        lowered = col.lower()
        if lowered in {"id", "station"} and "station_id" not in out.columns:
            rename[col] = "station_id"
        elif lowered == "station_id" and col != "station_id":
            rename[col] = "station_id"
        elif lowered == "latitude" and col != "latitude":
            rename[col] = "latitude"
        elif lowered == "longitude" and col != "longitude":
            rename[col] = "longitude"
        elif lowered == "length" and col != "length":
            rename[col] = "length"
    out = out.rename(columns=rename)
    if "station_id" not in out.columns:
        raise ValueError("Meta CSV must contain station_id or ID")
    if "latitude" not in out.columns or "longitude" not in out.columns:
        raise ValueError("Meta CSV must contain latitude and longitude columns")
    out["station_id"] = out["station_id"].astype(int)
    return out


def _load_graph_nodes(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        payload = payload.get("graph_nodes", payload)
    return [int(x) for x in np.asarray(payload).tolist()]


def _positive_finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value) and value > 0:
        return value
    return None


def _haversine_miles(lat1, lon1, lat2, lon2):
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in vals):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, map(float, vals))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    miles = 3958.7613 * c
    return _positive_finite(miles)


def _edge_length(meta_by_sid, src_sid, dst_sid, default_length):
    src = meta_by_sid.get(src_sid)
    dst = meta_by_sid.get(dst_sid)
    if src is not None and "length" in src:
        length = _positive_finite(src.get("length"))
        if length is not None:
            return length

    if src is not None and dst is not None:
        distance = _haversine_miles(
            src.get("latitude"),
            src.get("longitude"),
            dst.get("latitude"),
            dst.get("longitude"),
        )
        if distance is not None:
            return distance

    fallback = _positive_finite(default_length)
    if fallback is None:
        raise ValueError("--default-length must be a positive finite number")
    return fallback


def _derive_graph_nodes(meta):
    valid = meta.dropna(subset=["latitude", "longitude"]).copy()
    return sorted(valid["station_id"].astype(int).unique().tolist())


def _build_graph(edge_index, graph_nodes, meta, default_length, reverse_edge_index=True):
    if isinstance(edge_index, torch.Tensor):
        edge_np = edge_index.detach().cpu().numpy()
    else:
        edge_np = np.asarray(edge_index)
    if edge_np.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_np.shape}")

    meta_by_sid = {
        int(row["station_id"]): row
        for row in meta.to_dict(orient="records")
    }
    G = nx.DiGraph()
    G.add_nodes_from(range(len(graph_nodes)))

    for row, col in zip(edge_np[0], edge_np[1]):
        if reverse_edge_index:
            src_idx, dst_idx = int(col), int(row)
        else:
            src_idx, dst_idx = int(row), int(col)
        if src_idx >= len(graph_nodes) or dst_idx >= len(graph_nodes) or src_idx < 0 or dst_idx < 0:
            raise ValueError(
                "edge_index references node outside graph_nodes: "
                f"edge ({src_idx}, {dst_idx}), node count {len(graph_nodes)}"
            )
        src_sid = int(graph_nodes[src_idx])
        dst_sid = int(graph_nodes[dst_idx])
        G.add_edge(
            src_idx,
            dst_idx,
            length=_edge_length(meta_by_sid, src_sid, dst_sid, default_length),
        )
    return G


def _reachable_within(G, start, max_distance, default_length):
    visited = {start}
    best = {start: 0.0}
    queue = deque([(start, 0.0)])
    fallback = _positive_finite(default_length)
    if fallback is None:
        raise ValueError("--default-length must be a positive finite number")

    while queue:
        node, distance = queue.popleft()
        for nbr, attr in G[node].items():
            edge_len = _positive_finite(attr.get("length")) or fallback
            new_distance = distance + edge_len
            if new_distance <= max_distance and new_distance < best.get(nbr, math.inf):
                visited.add(nbr)
                best[nbr] = new_distance
                queue.append((nbr, new_distance))
    return sorted(visited)


def _static_edges_from_neighbors(neighbors):
    static_edges = {}
    for name, lists in neighbors.items():
        edge_list = [[src, dst] for src, nbrs in enumerate(lists) for dst in nbrs]
        if edge_list:
            static_edges[name] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            static_edges[name] = torch.empty((2, 0), dtype=torch.long)
    return static_edges


def precompute_neighbors(
    meta,
    edge_index,
    graph_nodes=None,
    default_length=DEFAULT_LENGTH,
    reverse_edge_index=True,
):
    meta = _normalise_meta(meta)
    if graph_nodes is None:
        graph_nodes = _derive_graph_nodes(meta)
    else:
        graph_nodes = [int(sid) for sid in graph_nodes]
    if not graph_nodes:
        raise ValueError("No graph nodes available for neighbor precomputation")

    G = _build_graph(
        edge_index=edge_index,
        graph_nodes=graph_nodes,
        meta=meta,
        default_length=default_length,
        reverse_edge_index=reverse_edge_index,
    )

    neighbors = {name: [] for name in HOURS}
    for node_idx in range(len(graph_nodes)):
        for name, max_distance in HOURS.items():
            neighbors[name].append(
                _reachable_within(G, node_idx, max_distance, default_length)
            )

    payload = {
        "graph_nodes": graph_nodes,
        "neighbors": neighbors,
    }
    return payload, _static_edges_from_neighbors(neighbors)


def _save_pickle(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _save_edge(path, edge_index):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(edge_index, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=_env("PEMS_BASE_DIR", DEFAULT_BASE))
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--edge-index-pt", default=None)
    parser.add_argument("--graph-nodes-pkl", default=None)
    parser.add_argument("--out-neighbors-pkl", default=None)
    parser.add_argument("--out-edge-5", default=None)
    parser.add_argument("--out-edge-15", default=None)
    parser.add_argument("--out-edge-30", default=None)
    parser.add_argument("--default-length", type=float, default=DEFAULT_LENGTH)
    parser.add_argument(
        "--use-edge-index-direction",
        action="store_true",
        help="Traverse row->col instead of preserving the previous col->row behavior.",
    )
    args = parser.parse_args(argv)

    base = args.base_dir
    args.meta_csv = _resolve_path(args.meta_csv, base, "step01_d07_meta.csv", "STEP62_META_CSV")
    args.edge_index_pt = _resolve_path(
        args.edge_index_pt, base, "step52_edge_index.pt", "STEP62_EDGE_INDEX_PT"
    )
    default_nodes = os.path.join(base, "step52_graph_nodes.pkl")
    args.graph_nodes_pkl = args.graph_nodes_pkl or _env("STEP62_GRAPH_NODES_PKL")
    if args.graph_nodes_pkl is None and os.path.exists(default_nodes):
        args.graph_nodes_pkl = default_nodes
    args.out_neighbors_pkl = _resolve_path(
        args.out_neighbors_pkl, base, "step62_neighbors.pkl", "STEP62_OUT_NEIGHBORS_PKL"
    )
    args.out_edge_5 = _resolve_path(args.out_edge_5, base, "edge_index_5min.pt", "STEP62_OUT_EDGE_5")
    args.out_edge_15 = _resolve_path(
        args.out_edge_15, base, "edge_index_15min.pt", "STEP62_OUT_EDGE_15"
    )
    args.out_edge_30 = _resolve_path(
        args.out_edge_30, base, "edge_index_30min.pt", "STEP62_OUT_EDGE_30"
    )
    return args


def main(argv=None):
    args = parse_args(argv)

    meta = pd.read_csv(args.meta_csv)
    edge_index = torch.load(args.edge_index_pt, map_location="cpu")
    graph_nodes = _load_graph_nodes(args.graph_nodes_pkl) if args.graph_nodes_pkl else None

    payload, static_edges = precompute_neighbors(
        meta=meta,
        edge_index=edge_index,
        graph_nodes=graph_nodes,
        default_length=args.default_length,
        reverse_edge_index=not args.use_edge_index_direction,
    )

    _save_pickle(args.out_neighbors_pkl, payload)
    _save_edge(args.out_edge_5, static_edges["5min"])
    _save_edge(args.out_edge_15, static_edges["15min"])
    _save_edge(args.out_edge_30, static_edges["30min"])

    node_count = len(payload["graph_nodes"])
    print(f"Precomputed neighbors for {node_count} nodes.")
    for name in HOURS:
        avg_size = sum(len(lst) for lst in payload["neighbors"][name]) / node_count
        print(f"  {name}: avg neighborhood size = {avg_size:.1f}")
    print(f"Saved neighbors to {args.out_neighbors_pkl}")
    print(f"Saved {args.out_edge_5}")
    print(f"Saved {args.out_edge_15}")
    print(f"Saved {args.out_edge_30}")


if __name__ == "__main__":
    main()
