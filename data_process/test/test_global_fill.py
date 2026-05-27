import os
import sys
import unittest
import numpy as np

# 把 data_process 目录加入到 sys.path，确保能 import interp_utils.py
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from interp_utils import global_fill  # 直接从同级目录导入

import pandas as pd

class TestGlobalFill(unittest.TestCase):
    def test_mixed_group(self):
        df = pd.DataFrame({
            'flow':       [1.0,   np.nan, 3.0,  np.nan],
            'occupancy':  [0.1,    np.nan, 0.3,  np.nan],
            'mask_flag':  [False, True,   False, True]
        })
        out = global_fill(df.copy(), ['flow', 'occupancy'])
        # 有效行平均 (flow: (1+3)/2=2, occ:(0.1+0.3)/2=0.2)
        self.assertAlmostEqual(out.loc[1, 'flow'], 2.0)
        self.assertAlmostEqual(out.loc[1, 'occupancy'], 0.2)
        self.assertAlmostEqual(out.loc[3, 'flow'], 2.0)
        self.assertAlmostEqual(out.loc[3, 'occupancy'], 0.2)

    def test_all_masked(self):
        df = pd.DataFrame({
            'flow':       [np.nan, np.nan],
            'occupancy':  [np.nan, np.nan],
            'mask_flag':  [True,   True]
        })
        out = global_fill(df.copy(), ['flow', 'occupancy'])
        self.assertTrue(np.isnan(out.loc[0, 'flow']))
        self.assertTrue(np.isnan(out.loc[1, 'occupancy']))

    def test_no_mask(self):
        df = pd.DataFrame({
            'flow':       [5.0, 10.0],
            'occupancy':  [0.5,  0.8],
            'mask_flag':  [False, False]
        })
        out = global_fill(df.copy(), ['flow', 'occupancy'])
        pd.testing.assert_frame_equal(df, out)

if __name__ == '__main__':
    unittest.main()
import os, sys, unittest
# 确保能 import data_process 下的 interp_utils
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from interp_utils import local_fill

class TestLocalFill(unittest.TestCase):
    def test_basic_fill(self):
        # 三行，邻居列表长度为 2
        df = pd.DataFrame({
            'flow':      [1.0, np.nan, 3.0],
            'nbr_idx':   [[0,2], [0,2], [0,2]],
            'mask_flag': [False, True, False]
        })
        out = local_fill(df, 'flow')
        # 行 1 应该被 (1 + 3) / 2 填充
        self.assertAlmostEqual(out.loc[1, 'flow'], 2.0)

    def test_no_valid_neighbors(self):
        # 三行，只有邻居是自己或都是 mask ⇒ 不变
        df = pd.DataFrame({
            'flow':      [np.nan, np.nan],
            'nbr_idx':   [[1], [0]],
            'mask_flag': [True, True]
        })
        out = local_fill(df, 'flow')
        self.assertTrue(np.isnan(out.loc[0, 'flow']))
        self.assertTrue(np.isnan(out.loc[1, 'flow']))

    def test_partial_nan_neighbors(self):
        # 混合一些 NaN 和 有效值
        df = pd.DataFrame({
            'flow':      [10.0, np.nan, 30.0, np.nan],
            'nbr_idx':   [[0,2], [0,2], [0,2], [0,2]],
            'mask_flag': [False, True, False, True]
        })
        out = local_fill(df, 'flow')
        # 行 1 和 行 3 都应该是 (10+30)/2 = 20
        self.assertAlmostEqual(out.loc[1, 'flow'], 20.0)
        self.assertAlmostEqual(out.loc[3, 'flow'], 20.0)

if __name__ == '__main__':
    unittest.main()
