#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a KNN topology over the sorted valid station list.

Defaults preserve the production /scratch paths. CLI arguments or environment
variables can redirect inputs and outputs for small local samples.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


DEFAULT_BASE = "/scratch/lgong1/finalproject/pems_data"
DEFAULT_K_NEIGHBORS = 5


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
    out = out.rename(columns=rename)
    if "station_id" not in out.columns:
        raise ValueError("Meta CSV must contain station_id or ID")
    if "latitude" not in out.columns or "longitude" not in out.columns:
        raise ValueError("Meta CSV must contain latitude and longitude columns")
    out["station_id"] = out["station_id"].astype(int)
    return out


def build_topology(station_ids, meta, k_neighbors=DEFAULT_K_NEIGHBORS):
    station_ids = sorted({int(sid) for sid in station_ids})
    meta = _normalise_meta(meta)

    meta = meta[meta["station_id"].isin(station_ids)].copy()
    meta = meta.sort_values("station_id").drop_duplicates("station_id")
    present = set(meta["station_id"].astype(int).tolist())
    missing_meta = sorted(set(station_ids) - present)

    coord_missing_mask = meta[["latitude", "longitude"]].isna().any(axis=1)
    dropped_missing_coords = meta.loc[coord_missing_mask, "station_id"].astype(int).tolist()
    valid = meta.loc[~coord_missing_mask].copy()
    graph_nodes = valid["station_id"].astype(int).tolist()

    if not graph_nodes:
        raise ValueError("No stations have valid coordinates for topology generation")

    k_neighbors = max(0, int(k_neighbors))
    if len(graph_nodes) == 1 or k_neighbors == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index, graph_nodes, dropped_missing_coords, missing_meta

    coords = valid[["latitude", "longitude"]].to_numpy(dtype=float)
    n_neighbors = min(k_neighbors + 1, len(graph_nodes))
    nbrs_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    nbrs_model.fit(coords)
    _, indices = nbrs_model.kneighbors(coords)

    src_list = []
    dst_list = []
    for src, neigh in enumerate(indices):
        added = 0
        for dst in neigh:
            dst = int(dst)
            if dst == src:
                continue
            src_list.append(src)
            dst_list.append(dst)
            added += 1
            if added >= k_neighbors:
                break

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index, graph_nodes, dropped_missing_coords, missing_meta


def _read_interp_station_ids(path):
    df = pd.read_parquet(path, columns=["station_id"])
    return df["station_id"].astype(int).unique().tolist()


def _save_node_mapping(path, graph_nodes, dropped_missing_coords, missing_meta):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "graph_nodes": graph_nodes,
        "station_to_node": {int(sid): idx for idx, sid in enumerate(graph_nodes)},
        "dropped_missing_coordinates": dropped_missing_coords,
        "missing_metadata": missing_meta,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=_env("PEMS_BASE_DIR", DEFAULT_BASE))
    parser.add_argument("--interp-parq", default=None)
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--out-edge-index", default=None)
    parser.add_argument("--out-graph-nodes-pkl", default=None)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    args = parser.parse_args(argv)

    base = args.base_dir
    args.interp_parq = _resolve_path(
        args.interp_parq, base, "step51_interpolated_final.parquet", "STEP52_INTERP_PARQ"
    )
    args.meta_csv = _resolve_path(args.meta_csv, base, "step01_d07_meta.csv", "STEP52_META_CSV")
    args.out_edge_index = _resolve_path(
        args.out_edge_index, base, "step52_edge_index.pt", "STEP52_OUT_EDGE_INDEX"
    )
    args.out_graph_nodes_pkl = _resolve_path(
        args.out_graph_nodes_pkl,
        base,
        "step52_graph_nodes.pkl",
        "STEP52_OUT_GRAPH_NODES_PKL",
    )
    return args


def main(argv=None):
    args = parse_args(argv)

    station_ids = _read_interp_station_ids(args.interp_parq)
    print(f"Stations in interpolated data: {len(station_ids)}")
    meta = pd.read_csv(args.meta_csv)
    edge_index, graph_nodes, dropped_missing_coords, missing_meta = build_topology(
        station_ids=station_ids,
        meta=meta,
        k_neighbors=args.k_neighbors,
    )

    if missing_meta:
        print(
            "Stations missing from metadata: "
            f"{missing_meta[:20]}{' ...' if len(missing_meta) > 20 else ''}"
        )
    if dropped_missing_coords:
        print(
            "Dropped station IDs with missing coordinates: "
            f"{dropped_missing_coords[:20]}{' ...' if len(dropped_missing_coords) > 20 else ''}"
        )
    else:
        print("Dropped station IDs with missing coordinates: []")

    parent = os.path.dirname(os.path.abspath(args.out_edge_index))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(edge_index, args.out_edge_index)
    _save_node_mapping(
        args.out_graph_nodes_pkl,
        graph_nodes,
        dropped_missing_coords,
        missing_meta,
    )

    print(
        f"Built edge_index with {edge_index.shape[1]} edges on "
        f"{len(graph_nodes)} sorted valid stations."
    )
    print(f"Saved edge_index to: {args.out_edge_index}")
    print(f"Saved graph node mapping to: {args.out_graph_nodes_pkl}")


if __name__ == "__main__":
    main()
