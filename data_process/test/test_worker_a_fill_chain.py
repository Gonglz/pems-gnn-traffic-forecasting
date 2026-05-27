import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestStep50LocalAlignment(unittest.TestCase):
    def test_local_fill_uses_neighbor_station_at_same_timestamp(self):
        from step50_Fill import apply_partition_local_fill, initialize_masked_features

        ts = pd.Timestamp('2024-01-01 00:00:00')
        df = pd.DataFrame({
            'timestamp': [ts, ts, ts + pd.Timedelta(minutes=5)],
            'station_id': [100, 200, 300],
            'direction': ['N', 'N', 'N'],
            'flow': [999.0, 20.0, 300.0],
            'occupancy': [9.99, 0.20, 3.00],
            'speed': [999.0, 55.0, 30.0],
            'mask_logic': [True, False, False],
            'mask_md': [False, False, False],
            'mask_hf': [False, False, False],
        })
        nbr_map = pd.DataFrame({
            'station_id': [100, 100],
            'neighbor_station_id': [200, 300],
        })

        prepared = initialize_masked_features(df)
        out = apply_partition_local_fill(prepared, nbr_map)

        filled = out.loc[out['station_id'] == 100].iloc[0]
        self.assertAlmostEqual(filled['flow'], 20.0)
        self.assertAlmostEqual(filled['occupancy'], 0.20)
        self.assertAlmostEqual(filled['speed'], 55.0)

    def test_local_fill_leaves_missing_when_neighbor_timestamp_absent(self):
        from step50_Fill import apply_partition_local_fill, initialize_masked_features

        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 00:00:00'),
                          pd.Timestamp('2024-01-01 00:05:00')],
            'station_id': [100, 200],
            'direction': ['N', 'N'],
            'flow': [999.0, 20.0],
            'occupancy': [9.99, 0.20],
            'speed': [999.0, 55.0],
            'mask_logic': [True, False],
            'mask_md': [False, False],
            'mask_hf': [False, False],
        })
        nbr_map = pd.DataFrame({
            'station_id': [100],
            'neighbor_station_id': [200],
        })

        prepared = initialize_masked_features(df)
        out = apply_partition_local_fill(prepared, nbr_map)

        filled = out.loc[out['station_id'] == 100].iloc[0]
        self.assertTrue(np.isnan(filled['flow']))
        self.assertTrue(np.isnan(filled['occupancy']))
        self.assertTrue(np.isnan(filled['speed']))


class TestStep51Refill(unittest.TestCase):
    def test_refill_does_not_use_raw_masked_values(self):
        from step51_refill import build_refilled_frame

        ts = pd.Timestamp('2024-01-01 00:00:00')
        raw = pd.DataFrame({
            'timestamp': [ts, ts],
            'station_id': [100, 200],
            'direction': ['N', 'N'],
            'flow': [999.0, 20.0],
            'occupancy': [9.99, 0.20],
            'speed': [999.0, 55.0],
        })
        mask = pd.DataFrame({
            'timestamp': [ts],
            'station_id': [100],
            'direction': ['N'],
            'mask_logic': [True],
            'mask_md': [False],
            'mask_hf': [False],
        })
        interp = pd.DataFrame({
            'timestamp': [ts, ts],
            'station_id': [100, 200],
            'direction': ['N', 'N'],
            'flow': [np.nan, 20.0],
            'occupancy': [np.nan, 0.20],
            'speed': [np.nan, 55.0],
        })

        out = build_refilled_frame(raw, mask, interp)
        row = out.loc[out['station_id'] == 100].iloc[0]

        self.assertAlmostEqual(row['flow_interp'], 20.0)
        self.assertAlmostEqual(row['occupancy_interp'], 0.20)
        self.assertAlmostEqual(row['speed_interp'], 55.0)

    def test_refill_collapses_duplicates_to_expected_unique_count(self):
        from step51_refill import build_refilled_frame

        ts = pd.Timestamp('2024-01-01 00:00:00')
        raw = pd.DataFrame({
            'timestamp': [ts, ts, ts],
            'station_id': [100, 100, 200],
            'direction': ['N', 'N', 'N'],
        })
        mask = pd.DataFrame({
            'timestamp': [ts, ts],
            'station_id': [100, 100],
            'direction': ['N', 'N'],
            'mask_logic': [True, True],
            'mask_md': [False, False],
            'mask_hf': [False, False],
        })
        interp = pd.DataFrame({
            'timestamp': [ts, ts, ts],
            'station_id': [100, 200, 200],
            'direction': ['N', 'N', 'N'],
            'flow_interp': [np.nan, 20.0, 24.0],
            'occupancy_interp': [np.nan, 0.20, 0.24],
            'speed_interp': [np.nan, 50.0, 54.0],
        })

        out = build_refilled_frame(raw, mask, interp)

        self.assertEqual(len(out), 2)
        row = out.loc[out['station_id'] == 100].iloc[0]
        self.assertAlmostEqual(row['flow_interp'], 22.0)
        self.assertAlmostEqual(row['occupancy_interp'], 0.22)
        self.assertAlmostEqual(row['speed_interp'], 52.0)


class TestFillCheck(unittest.TestCase):
    def test_fill_check_accepts_step50_and_step51_schemas(self):
        from step50_fillCheck import compute_check_stats

        ts = pd.Timestamp('2024-01-01 00:00:00')
        raw = pd.DataFrame({
            'timestamp': [ts, ts],
            'station_id': [100, 200],
            'flow': [10.0, 20.0],
            'occupancy': [0.10, 0.20],
            'speed': [50.0, 55.0],
        })
        mask = pd.DataFrame({
            'timestamp': [ts],
            'station_id': [100],
            'mask_logic': [True],
            'mask_md': [False],
            'mask_hf': [False],
        })
        step50_interp = pd.DataFrame({
            'timestamp': [ts, ts],
            'station_id': [100, 200],
            'flow': [11.0, 20.0],
            'occupancy': [0.11, 0.20],
            'speed': [51.0, 55.0],
        })
        step51_interp = step50_interp.rename(columns={
            'flow': 'flow_interp',
            'occupancy': 'occupancy_interp',
            'speed': 'speed_interp',
        })

        stats50 = compute_check_stats(raw, mask, step50_interp)
        stats51 = compute_check_stats(raw, mask, step51_interp)

        self.assertEqual(stats50['missing_masked_count'], 0)
        self.assertEqual(stats51['missing_masked_count'], 0)
        self.assertEqual(stats50['unmasked_max_diffs']['flow'], 0.0)
        self.assertEqual(stats51['unmasked_max_diffs']['speed'], 0.0)


if __name__ == '__main__':
    unittest.main()
