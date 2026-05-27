import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import torch_geometric  # noqa: F401
except ModuleNotFoundError:
    tg = types.ModuleType("torch_geometric")
    tg_data = types.ModuleType("torch_geometric.data")
    tg_loader = types.ModuleType("torch_geometric.loader")
    tg_nn = types.ModuleType("torch_geometric.nn")

    class InMemoryDataset:
        def __init__(self, root=None):
            self.root = root

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class NeighborSampler:
        pass

    class SAGEConv(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    tg_data.InMemoryDataset = InMemoryDataset
    tg_data.Data = Data
    tg_loader.NeighborSampler = NeighborSampler
    tg_nn.SAGEConv = SAGEConv
    sys.modules["torch_geometric"] = tg
    sys.modules["torch_geometric.data"] = tg_data
    sys.modules["torch_geometric.loader"] = tg_loader
    sys.modules["torch_geometric.nn"] = tg_nn

from model.dataset_full import RFGraphDatasetFull
from model.predict_and_plot import apply_realtime_overrides, resolve_prediction_node
from model.gnn_final import MultiHeadRFMLP
from model.p2_quality import (
    apply_input_flow_normalization_,
    apply_temporal_encoding_,
    audit_forecast_task,
    compute_stationwise_flow_stats,
    inverse_transform_prediction,
    temporal_feature_matrix,
)
from model.train_sampler import (
    build_time_splits,
    count_node_batches,
    flat_batch_work_items,
    partition_indices_for_rank,
    rank_seed_nodes,
)
from model.training_modes import apply_feature_ablation_, apply_graph_mode


def _write_tiny_dataset(root, old_neighbor_payload=False):
    root.mkdir(parents=True, exist_ok=True)
    x = np.zeros((10, 3, 6), dtype=np.float32)
    y = np.zeros((10, 3), dtype=np.float32)
    x[:, 0, 0] = 100.0
    x[:, 1, 0] = 200.0
    x[:, 2, 0] = 300.0
    y[:, 0] = 10.0
    y[:, 1] = 20.0
    y[:, 2] = 30.0
    np.save(root / "X_ext.npy", x)
    np.save(root / "Y.npy", y)
    np.save(root / "sids.npy", np.array([10, 20, 30], dtype=np.int64))
    np.save(
        root / "timestamps.npy",
        np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:05:00",
                "2024-01-01T00:10:00",
                "2024-01-01T00:15:00",
                "2024-01-01T00:20:00",
                "2024-01-01T00:25:00",
                "2024-01-01T00:30:00",
                "2024-01-01T00:35:00",
                "2024-01-01T00:40:00",
                "2024-01-01T00:45:00",
            ],
            dtype="datetime64[s]",
        ),
    )
    if old_neighbor_payload:
        neighbors = {
            "5min": [[1], [0, 2], [1]],
            "15min": [[1], [0, 2], [1]],
            "30min": [[1], [0, 2], [1]],
        }
        payload = neighbors
    else:
        neighbors = {
            "5min": [[1], [0]],
            "15min": [[1], [0]],
            "30min": [[1], [0]],
        }
        payload = {
            "graph_nodes": [30, 10],
            "neighbors": neighbors,
        }
    with open(root / "step62_neighbors.pkl", "wb") as f:
        pickle.dump(payload, f)


class WorkerCTest(unittest.TestCase):
    def _dataset(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _write_tiny_dataset(root)
        return RFGraphDatasetFull(data_dir=str(root))

    def test_dataset_reorders_columns_to_graph_node_order_and_maps_ids(self):
        ds = self._dataset()

        self.assertEqual(ds.station_ids.tolist(), [30, 10])
        self.assertEqual(ds.station_id_to_node_idx(30), 0)
        self.assertEqual(ds.station_id_to_node_idx(10), 1)
        self.assertEqual(ds.node_idx_to_station_id(0), 30)
        self.assertEqual(ds.X[0, :, 0].tolist(), [300.0, 100.0])
        self.assertEqual(ds.Y[0, :].tolist(), [30.0, 10.0])

    def test_dataset_accepts_old_neighbor_payload_without_graph_nodes(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _write_tiny_dataset(root, old_neighbor_payload=True)

        ds = RFGraphDatasetFull(data_dir=str(root))

        self.assertEqual(ds.station_ids.tolist(), [10, 20, 30])

    def test_dataset_loads_timestamps_and_resolves_string_time(self):
        ds = self._dataset()

        self.assertEqual(ds.time_to_idx("2024-01-01 00:10:00"), 2)
        self.assertEqual(ds.time_to_idx(np.datetime64("2024-01-01T00:20:00")), 4)
        with self.assertRaises(KeyError):
            ds.time_to_idx("2024-01-02 00:00:00")

    def test_prediction_accepts_station_id_or_node_idx_and_feature_order(self):
        ds = self._dataset()
        features = torch.zeros((ds.N, ds.F), dtype=torch.float32)

        self.assertEqual(resolve_prediction_node(ds, station_id=10, node_idx=None), 1)
        self.assertEqual(resolve_prediction_node(ds, station_id=None, node_idx=0), 0)
        with self.assertRaises(ValueError):
            resolve_prediction_node(ds, station_id=10, node_idx=0)

        apply_realtime_overrides(features, 1, flow=12.5, occupancy=0.33, speed=55.0)

        self.assertAlmostEqual(features[1, 0].item(), 12.5)
        self.assertAlmostEqual(features[1, 1].item(), 0.33)
        self.assertAlmostEqual(features[1, 2].item(), 55.0)

    def test_time_splits_are_disjoint_strided_and_limited(self):
        train_times, val_times = build_time_splits(
            total_times=21,
            max_horizon=6,
            time_stride=2,
            max_train_times=3,
            max_val_times=2,
        )

        self.assertEqual(train_times, [0, 2, 4])
        self.assertEqual(val_times, [12, 14])
        self.assertTrue(set(train_times).isdisjoint(val_times))

    def test_rank_partitioning_assigns_non_identical_work(self):
        items = list(range(10))

        rank0 = partition_indices_for_rank(items, rank=0, world_size=2)
        rank1 = partition_indices_for_rank(items, rank=1, world_size=2)
        nodes0 = rank_seed_nodes(5, rank=0, world_size=2).tolist()
        nodes1 = rank_seed_nodes(5, rank=1, world_size=2).tolist()

        self.assertNotEqual(rank0, rank1)
        self.assertEqual(sorted(rank0 + rank1), items)
        self.assertEqual(nodes0, [0, 2, 4])
        self.assertEqual(nodes1, [1, 3])

    def test_flat_batch_sharding_covers_time_node_batches_without_overlap(self):
        times = [10, 20, 30]
        num_batches = count_node_batches(num_nodes=5, batch_size=2)

        rank0 = flat_batch_work_items(times, num_batches, rank=0, world_size=2)
        rank1 = flat_batch_work_items(times, num_batches, rank=1, world_size=2)
        all_items = [(time_idx, batch_idx) for time_idx in times for batch_idx in range(num_batches)]

        self.assertEqual(sorted(rank0 + rank1), sorted(all_items))
        self.assertTrue(set(rank0).isdisjoint(rank1))

    def test_without_weather_ablation_uses_train_split_mean_only(self):
        x = torch.zeros((4, 2, 6), dtype=torch.float32)
        x[:, :, 3] = torch.tensor([[1.0, 3.0], [5.0, 7.0], [100.0, 100.0], [200.0, 200.0]])
        x[:, :, 4] = torch.tensor([[2.0, 4.0], [6.0, 8.0], [100.0, 100.0], [200.0, 200.0]])
        x[:, :, 0] = 42.0

        meta = apply_feature_ablation_(x, train_times=[0, 1], feature_ablation="without_weather")

        self.assertEqual(meta["weather_mean"], [4.0, 5.0])
        self.assertTrue(torch.allclose(x[:, :, 3], torch.full((4, 2), 4.0)))
        self.assertTrue(torch.allclose(x[:, :, 4], torch.full((4, 2), 5.0)))
        self.assertTrue(torch.allclose(x[:, :, 0], torch.full((4, 2), 42.0)))

    def test_without_graph_neighbors_uses_self_loop_edge_index(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        self_loop = apply_graph_mode(edge_index, num_nodes=3, graph_mode="self_loop")

        self.assertEqual(self_loop.dtype, torch.long)
        self.assertEqual(self_loop.tolist(), [[0, 1, 2], [0, 1, 2]])

    def test_task_audit_catches_future_target_leakage(self):
        ds = self._dataset()
        ds.X[:, :, 0] = 0.0
        ds.Y[:, :] = torch.arange(ds.T, dtype=torch.float32).unsqueeze(1)
        ds.X[:-1, :, 0] = ds.Y[1:, :]

        with self.assertRaises(ValueError):
            audit_forecast_task(ds, train_times=[0, 1])

    def test_stationwise_normalization_train_only_and_inverse(self):
        x = torch.zeros((5, 2, 6), dtype=torch.float32)
        y = torch.zeros((5, 2), dtype=torch.float32)
        x[:, :, 0] = torch.tensor([[1.0, 10.0], [3.0, 14.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        y[:, :] = torch.tensor([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0], [4.0, 16.0], [5.0, 18.0]])

        stats = compute_stationwise_flow_stats(x, y, train_times=[0, 1], head_deltas={"5min": 1})
        self.assertTrue(torch.allclose(stats["input_flow_mean"], torch.tensor([2.0, 12.0])))
        self.assertTrue(torch.allclose(stats["target_mean"]["5min"], torch.tensor([2.5, 13.0])))

        raw_pred = torch.tensor([[0.0], [1.0]])
        restored = inverse_transform_prediction(raw_pred, stats, "5min", torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(restored, torch.tensor([[2.5], [14.0]])))
        apply_input_flow_normalization_(x, stats)
        self.assertTrue(torch.allclose(x[0, :, 0], torch.tensor([-1.0, -1.0])))

    def test_temporal_features_are_deterministic_and_appended(self):
        ds = self._dataset()
        feats_a = temporal_feature_matrix(ds.timestamps)
        feats_b = temporal_feature_matrix(ds.timestamps)

        np.testing.assert_allclose(feats_a, feats_b)
        old_f = ds.F
        meta = apply_temporal_encoding_(ds, enabled=True)
        self.assertEqual(ds.F, old_f + 5)
        self.assertIn("sin_time_of_day", meta["feature_order"])

    def test_mlp_station_embedding_preserves_prediction_shape(self):
        model = MultiHeadRFMLP(
            in_dim=4,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            num_nodes=3,
            station_embedding_dim=2,
        )
        x = torch.zeros((2, 4), dtype=torch.float32)
        out = model(x=x, head="5min", node_ids=torch.tensor([0, 2]))

        self.assertEqual(tuple(out.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
