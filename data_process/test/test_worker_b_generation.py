import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FINALPROJECT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStep70WeatherFeatures(unittest.TestCase):
    def test_build_xy_arrays_aligns_weather_by_station_and_graph_order(self):
        mod = load_module(
            "step70_make_xy_weather",
            FINALPROJECT / "pems_data" / "step70_make_xy_weather.py",
        )
        timestamps = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:05"])
        interp = pd.DataFrame(
            {
                "timestamp": [timestamps[0], timestamps[0], timestamps[0], timestamps[1], timestamps[1], timestamps[1]],
                "station_id": [1, 2, 3, 1, 2, 3],
                "flow_interp": [10, 20, 30, 11, 21, 31],
                "occupancy_interp": [0.1, 0.2, 0.3, 0.11, 0.21, 0.31],
                "speed_interp": [50, 60, 70, 51, 61, 71],
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": [timestamps[0], timestamps[0], timestamps[1], timestamps[1]],
                "station_id": [1, 2, 1, 2],
                "tavg": [101, 202, 111, 212],
                "pcpn": [1.1, 2.2, 1.3, 2.4],
            }
        )

        X, Y, sids, out_ts = mod.build_xy_arrays(interp, weather, station_order=[2, 1])

        np.testing.assert_array_equal(sids, np.array([2, 1]))
        np.testing.assert_array_equal(out_ts, timestamps.to_numpy(dtype="datetime64[ns]"))
        self.assertEqual(X.shape, (2, 2, 6))
        self.assertEqual(Y.dtype, np.float32)
        np.testing.assert_allclose(X[:, 0, 3], [202, 212])
        np.testing.assert_allclose(X[:, 1, 3], [101, 111])
        np.testing.assert_allclose(X[:, 0, 4], [2.2, 2.4])
        np.testing.assert_allclose(Y[:, 0], [20, 21])

    def test_grid_weather_requires_station_grid_mapping(self):
        mod = load_module(
            "step70_make_xy_weather",
            FINALPROJECT / "pems_data" / "step70_make_xy_weather.py",
        )
        timestamp = pd.Timestamp("2025-01-01 00:00")
        interp = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "station_id": [1],
                "flow_interp": [10],
                "occupancy_interp": [0.1],
                "speed_interp": [50],
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "grid_id": ["grid_1"],
                "tavg": [99],
                "pcpn": [0.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "grid_id.*station_id.*mapping"):
            mod.build_xy_arrays(interp, weather)


class TestStep52Topology(unittest.TestCase):
    def test_build_topology_sorts_valid_station_ids_and_reports_missing_coordinates(self):
        mod = load_module(
            "step52_buildTopo",
            FINALPROJECT / "data_process" / "step52_buildTopo.py",
        )
        meta = pd.DataFrame(
            {
                "station_id": [3, 1, 2],
                "latitude": [37.2, 37.0, np.nan],
                "longitude": [-122.2, -122.0, -122.1],
            }
        )

        edge_index, graph_nodes, dropped, missing_meta = mod.build_topology(
            station_ids=[3, 1, 2],
            meta=meta,
            k_neighbors=1,
        )

        self.assertEqual(graph_nodes, [1, 3])
        self.assertEqual(dropped, [2])
        self.assertEqual(missing_meta, [])
        self.assertEqual(tuple(edge_index.shape), (2, 2))
        self.assertLessEqual(int(edge_index.max().item()), 1)


class TestStep62Neighbors(unittest.TestCase):
    def test_import_has_no_output_side_effects_and_nan_length_uses_coordinate_fallback(self):
        mod = load_module(
            "step62_precompute_neighbors",
            FINALPROJECT / "data_process" / "step62_precompute_neighbors.py",
        )
        meta = pd.DataFrame(
            {
                "station_id": [1, 2],
                "latitude": [37.0, 37.0005],
                "longitude": [-122.0, -122.0],
                "length": [np.nan, np.nan],
            }
        )
        edge_index = torch.tensor([[1], [0]], dtype=torch.long)

        payload, static_edges = mod.precompute_neighbors(
            meta=meta,
            edge_index=edge_index,
            graph_nodes=[1, 2],
        )

        self.assertIn(1, payload["neighbors"]["5min"][0])
        self.assertIn("5min", static_edges)
        self.assertEqual(tuple(static_edges["5min"].shape[0:1]), (2,))


if __name__ == "__main__":
    unittest.main()
