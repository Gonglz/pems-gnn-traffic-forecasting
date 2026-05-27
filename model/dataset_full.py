#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFGraphDatasetFull - dynamic RF neighbor dataset with station/time mappings.
"""

import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# Paths
DATA_DIR = "/scratch/lgong1/finalproject/pems_data"
STEP70_X = os.path.join(DATA_DIR, "X_ext.npy")
STEP70_Y = os.path.join(DATA_DIR, "Y.npy")
NEIGHBOR_PKL = os.path.join(DATA_DIR, "step62_neighbors.pkl")
SIDS_NPY = os.path.join(DATA_DIR, "sids.npy")
TIMESTAMPS_NPY = os.path.join(DATA_DIR, "timestamps.npy")

# Time deltas in 5-minute steps.
DELTA5, DELTA15, DELTA30 = 1, 3, 6


class RFGraphDatasetFull(InMemoryDataset):
    def __init__(
        self,
        data_dir=DATA_DIR,
        x_path=None,
        y_path=None,
        neighbor_path=None,
        sids_path=None,
        timestamps_path=None,
    ):
        super().__init__(data_dir)

        self.data_dir = data_dir
        self.x_path = x_path or os.path.join(data_dir, "X_ext.npy")
        self.y_path = y_path or os.path.join(data_dir, "Y.npy")
        self.neighbor_path = neighbor_path or os.path.join(data_dir, "step62_neighbors.pkl")
        self.sids_path = sids_path or os.path.join(data_dir, "sids.npy")
        default_timestamps_path = os.path.join(data_dir, "timestamps.npy")
        self.timestamps_path = timestamps_path or default_timestamps_path

        x = np.load(self.x_path)
        y = np.load(self.y_path)
        sids = np.asarray(np.load(self.sids_path), dtype=np.int64)
        with open(self.neighbor_path, "rb") as f:
            payload = pickle.load(f)

        if "graph_nodes" in payload:
            graph_nodes = np.asarray(payload["graph_nodes"], dtype=np.int64)
            nbrs = payload["neighbors"]
        else:
            # Backward compatibility for older neighbor pickles that stored only
            # {"5min": ..., "15min": ..., "30min": ...}. In that format the
            # node order is implicitly sids.npy.
            graph_nodes = sids
            nbrs = payload
        sid_to_col = {int(sid): idx for idx, sid in enumerate(sids.tolist())}
        kept_old_indices = []
        kept_station_ids = []
        col_indices = []
        for old_idx, sid in enumerate(graph_nodes.tolist()):
            if int(sid) in sid_to_col:
                kept_old_indices.append(old_idx)
                kept_station_ids.append(int(sid))
                col_indices.append(sid_to_col[int(sid)])

        if not kept_station_ids:
            raise ValueError("No graph nodes from step62_neighbors.pkl were found in sids.npy")

        x = x[:, col_indices, :]
        y = y[:, col_indices]

        self.station_ids = np.asarray(kept_station_ids, dtype=np.int64)
        self.graph_nodes = self.station_ids
        self._station_id_to_node_idx = {
            int(station_id): idx for idx, station_id in enumerate(self.station_ids.tolist())
        }

        self.T, self.N, self.F = x.shape
        self.X = torch.from_numpy(x).float()
        self.X_ext = self.X
        self.Y = torch.from_numpy(y).float()

        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_old_indices)}
        self.nbr5 = self._pad(self._remap_neighbors(nbrs["5min"], kept_old_indices, old_to_new))
        self.nbr15 = self._pad(self._remap_neighbors(nbrs["15min"], kept_old_indices, old_to_new))
        self.nbr30 = self._pad(self._remap_neighbors(nbrs["30min"], kept_old_indices, old_to_new))

        self.timestamps = self._load_timestamps(self.timestamps_path)
        self._time_to_idx = self._build_time_index(self.timestamps)

        self.train_list = []
        self.val_list = []
        maxd = max(DELTA5, DELTA15, DELTA30)
        t_eff = max(0, self.T - maxd)
        split = int(0.8 * t_eff)

        for t in range(t_eff):
            data = Data(
                x=self.X[t],
                nbr5=self.nbr5,
                nbr15=self.nbr15,
                nbr30=self.nbr30,
                y5=self.Y[t + DELTA5].unsqueeze(-1),
                y15=self.Y[t + DELTA15].unsqueeze(-1),
                y30=self.Y[t + DELTA30].unsqueeze(-1),
            )
            if t < split:
                self.train_list.append(data)
            else:
                self.val_list.append(data)

    def station_id_to_node_idx(self, station_id):
        station_id = int(station_id)
        if station_id not in self._station_id_to_node_idx:
            raise KeyError(f"Station ID {station_id} is not present in the graph")
        return self._station_id_to_node_idx[station_id]

    def node_idx_to_station_id(self, node_idx):
        node_idx = int(node_idx)
        if node_idx < 0 or node_idx >= self.N:
            raise IndexError(f"Node index {node_idx} is outside [0, {self.N})")
        return int(self.station_ids[node_idx])

    def time_to_idx(self, timestamp):
        if self.timestamps is None:
            raise ValueError(f"No timestamps.npy found at {self.timestamps_path}")
        key = self._time_key(timestamp)
        if key not in self._time_to_idx:
            raise KeyError(f"Timestamp {timestamp!r} is not present in timestamps.npy")
        return self._time_to_idx[key]

    def idx_to_time(self, time_idx):
        if self.timestamps is None:
            raise ValueError(f"No timestamps.npy found at {self.timestamps_path}")
        time_idx = int(time_idx)
        if time_idx < 0 or time_idx >= len(self.timestamps):
            raise IndexError(f"Time index {time_idx} is outside [0, {len(self.timestamps)})")
        return self.timestamps[time_idx]

    def _load_timestamps(self, timestamps_path):
        if not timestamps_path or not os.path.exists(timestamps_path):
            return None
        timestamps = np.load(timestamps_path)
        if len(timestamps) != self.T:
            raise ValueError(
                f"timestamps.npy has length {len(timestamps)}, but X/Y have T={self.T}"
            )
        return timestamps

    def _build_time_index(self, timestamps):
        if timestamps is None:
            return {}
        return {self._time_key(value): idx for idx, value in enumerate(timestamps)}

    def _time_key(self, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")
        try:
            return np.datetime_as_string(np.datetime64(value).astype("datetime64[s]"), unit="s")
        except (TypeError, ValueError):
            return str(value)

    def _remap_neighbors(self, lists, kept_old_indices, old_to_new):
        remapped = []
        for old_idx in kept_old_indices:
            converted = []
            for neighbor in lists[old_idx]:
                neighbor = int(neighbor)
                if neighbor in old_to_new:
                    converted.append(old_to_new[neighbor])
            remapped.append(converted)
        return remapped

    def _pad(self, lists):
        n = len(lists)
        k = max((len(lst) for lst in lists), default=0)
        mat = torch.full((n, k), -1, dtype=torch.long)
        for i, lst in enumerate(lists):
            if lst:
                mat[i, : len(lst)] = torch.tensor(lst, dtype=torch.long)
        return mat

    def len(self):
        return len(self.train_list)

    def get(self, idx):
        return self.train_list[idx]
