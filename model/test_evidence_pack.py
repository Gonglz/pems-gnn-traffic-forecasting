import csv
import json
import pickle
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from model.evidence_data import create_data_slice
from model.evidence_eval import historical_average_baseline, write_ablation_metrics, write_scaling_metrics
from model.evidence_pack import evaluate_available_checkpoints, write_run_metadata
from model.evidence_utils import compute_metric_row, split_time_indices


def write_tiny_source(root):
    root.mkdir(parents=True, exist_ok=True)
    x = np.zeros((10, 4, 6), dtype=np.float32)
    y = np.zeros((10, 4), dtype=np.float32)
    for t in range(10):
        for n in range(4):
            x[t, n, 0] = 100 * n + t
            y[t, n] = 100 * n + t
    np.save(root / "X_ext.npy", x)
    np.save(root / "Y.npy", y)
    np.save(root / "sids.npy", np.array([10, 20, 30, 40], dtype=np.int64))
    np.save(
        root / "timestamps.npy",
        np.array([f"2025-01-01T00:{i * 5:02d}:00" for i in range(10)], dtype="datetime64[s]"),
    )
    payload = {
        "graph_nodes": [30, 10],
        "neighbors": {
            "5min": [[1], [0]],
            "15min": [[1], [0]],
            "30min": [[1], [0]],
        },
    }
    with open(root / "step62_neighbors.pkl", "wb") as f:
        pickle.dump(payload, f)
    torch.save(torch.tensor([[0, 1], [1, 0]], dtype=torch.long), root / "step52_edge_index.pt")
    (root / "step01_d07_meta.csv").write_text(
        "station_id,latitude,longitude\n10,34.0,-118.0\n30,34.1,-118.1\n"
    )


class EvidencePackTest(unittest.TestCase):
    def test_create_data_slice_preserves_graph_node_order_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source"
            out = tmp / "evidence"
            write_tiny_source(source)

            payload = create_data_slice(
                source_data_dir=source,
                output_root=out,
                all_times=True,
                meta_csv=source / "step01_d07_meta.csv",
            )

            self.assertEqual(payload["time_steps"], 10)
            np.testing.assert_array_equal(np.load(out / "pems_data" / "sids.npy"), np.array([30, 10]))
            x = np.load(out / "pems_data" / "X_ext.npy")
            y = np.load(out / "pems_data" / "Y.npy")
            self.assertEqual(x.shape, (10, 2, 6))
            self.assertEqual(y.dtype, np.float32)
            self.assertEqual(x[0, :, 0].tolist(), [200.0, 0.0])
            self.assertTrue((out / "pems_data" / "timestamps.npy").exists())
            self.assertTrue((out / "config" / "data_slice.json").exists())

    def test_metrics_formula(self):
        row = compute_metric_row(
            pred=np.array([2.0, 4.0]),
            true=np.array([1.0, 2.0]),
            run_id="run",
            split="test",
            model="m",
            ablation="none",
            horizon_min=5,
        )

        self.assertAlmostEqual(row["mae"], 1.5)
        self.assertAlmostEqual(row["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(row["mape"], 100.0)
        self.assertEqual(row["n_samples"], 2)

    def test_historical_average_uses_train_only_slots(self):
        y = np.zeros((12, 2), dtype=np.float32)
        y[:, 0] = np.arange(12)
        y[:, 1] = np.arange(12) + 100
        timestamps = np.array(
            [f"2025-01-01T00:{i * 5:02d}:00" for i in range(12)],
            dtype="datetime64[s]",
        )
        accs = historical_average_baseline(
            y,
            timestamps,
            train_times=np.array([0, 1, 2, 3]),
            test_times=np.array([4]),
        )

        self.assertGreater(accs["5min"].n, 0)
        self.assertGreater(accs["15min"].n, 0)
        self.assertGreater(accs["30min"].n, 0)

    def test_scaling_metrics_speedup_from_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, phase_time, peak in [
                ("single_gpu", "10.0", 100),
                ("ddp_2gpu", "6.0", 120),
            ]:
                ckpt = root / "checkpoints" / rel
                ckpt.mkdir(parents=True)
                with open(ckpt / "epoch_metrics.csv", "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "phase",
                            "mse",
                            "steps",
                            "phase_time_sec",
                            "elapsed_sec",
                            "world_size",
                            "peak_gpu_mem_mb",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "epoch": 1,
                            "phase": "train",
                            "mse": 1,
                            "steps": 1,
                            "phase_time_sec": phase_time,
                            "elapsed_sec": phase_time,
                            "world_size": 1,
                            "peak_gpu_mem_mb": peak,
                        }
                    )
                with open(ckpt / "train_config.json", "w") as f:
                    json.dump({"total_time_sec": float(phase_time), "peak_gpu_mem_mb": peak}, f)

            rows = write_scaling_metrics(root)

            self.assertEqual(rows[0]["speedup"], 1.0)
            self.assertAlmostEqual(rows[1]["speedup"], 10.0 / 6.0)
            self.assertTrue((root / "metrics" / "scaling_metrics.csv").exists())

    def test_scaling_metrics_marks_skipped_single_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir(parents=True)
            (root / "logs" / "single_gpu.log").write_text("single GPU training skipped by --skip-single\n")
            ckpt = root / "checkpoints" / "ddp_2gpu"
            ckpt.mkdir(parents=True)
            with open(ckpt / "epoch_metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "phase": "train",
                        "mse": 1,
                        "steps": 2,
                        "pairs": 8,
                        "pairs_per_sec": 4,
                        "phase_time_sec": 2.0,
                        "elapsed_sec": 2.0,
                        "world_size": 2,
                        "peak_gpu_mem_mb": 120,
                    }
                )
            with open(ckpt / "train_config.json", "w") as f:
                json.dump({"total_time_sec": 2.0, "peak_gpu_mem_mb": 120, "ddp_shard_mode": "flat-batch"}, f)

            rows = write_scaling_metrics(root)

            self.assertEqual(rows[0]["notes"], "skipped_by_config")
            self.assertEqual(rows[1]["speedup"], "")
            self.assertEqual(rows[1]["notes"], "ddp_shard_mode=flat-batch")

    def test_ablation_metrics_marks_without_graph_unavailable_when_rf_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics"
            metrics.mkdir(parents=True)
            with open(metrics / "baseline_metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "run",
                        "split": "test",
                        "model": "last_value",
                        "ablation": "none",
                        "horizon_min": "all",
                        "mae": 1,
                        "rmse": 1,
                        "mape": 1,
                        "n_samples": 1,
                        "notes": "naive temporal",
                    }
                )

            rows = write_ablation_metrics(root)

            missing = [row for row in rows if row["ablation"] == "without_graph_neighbors"]
            self.assertEqual(len(missing), 4)
            self.assertTrue(all(row["notes"] == "not_available" for row in missing))

    def test_write_run_metadata_creates_environment_and_code_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(python_bin=sys.executable, code_ref="/tmp/code-backup.tar.gz")

            write_run_metadata(root, args)

            self.assertIn("Python", (root / "config" / "environment.txt").read_text())
            self.assertEqual((root / "config" / "git_or_backup_ref.txt").read_text().strip(), "/tmp/code-backup.tar.gz")

    def test_evaluate_available_checkpoints_writes_single_and_ddp_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "pems_data"
            data_dir.mkdir(parents=True)
            np.save(data_dir / "sids.npy", np.array([10], dtype=np.int64))
            for rel in ["single_gpu", "ddp_2gpu"]:
                ckpt = root / "checkpoints" / rel
                ckpt.mkdir(parents=True)
                (ckpt / "best_rf_gnn_dynamic_sampler.pth").write_text("placeholder")
                with open(ckpt / "train_config.json", "w") as f:
                    json.dump({"total_time_sec": 1.0, "peak_gpu_mem_mb": 2.0}, f)
            args = SimpleNamespace(
                checkpoint_path=None,
                plot_station_id=None,
                max_eval_times=1,
                feature_ablation="none",
                graph_mode="full",
            )

            def fake_eval(**kwargs):
                return [
                    {
                        "run_id": kwargs["run_id"],
                        "split": "test",
                        "model": kwargs["model_name"],
                        "ablation": "full",
                        "horizon_min": "all",
                        "mae": 1.0,
                        "rmse": 1.0,
                        "mape": 1.0,
                        "n_samples": 1,
                        "train_time_sec": kwargs["train_time_sec"],
                        "peak_gpu_mem_mb": kwargs["peak_gpu_mem_mb"],
                        "notes": kwargs["notes"],
                    }
                ]

            with patch("model.evidence_pack.evaluate_checkpoint_metrics", side_effect=fake_eval):
                rows = evaluate_available_checkpoints(args, data_dir, root)

            self.assertEqual([row["model"] for row in rows], ["graphsage_weather_single_gpu", "graphsage_weather_ddp_2gpu"])
            self.assertTrue((root / "metrics" / "model_metrics.csv").exists())

    def test_split_time_indices_has_test_split(self):
        splits = split_time_indices(20)

        self.assertTrue(len(splits["train"]) > 0)
        self.assertTrue(len(splits["val"]) > 0)
        self.assertTrue(len(splits["test"]) > 0)
        self.assertLess(max(splits["train"]), min(splits["val"]))
        self.assertLess(max(splits["val"]), min(splits["test"]))


if __name__ == "__main__":
    unittest.main()
