import os, sys, unittest
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from interp_utils import temporal_fill

class TestTemporalFill(unittest.TestCase):
    def setUp(self):
        # 构造简单时间序列：timestamp 用 0,1,2,3
        self.base = pd.DataFrame({
            'station_id': [1,1,1,1],
            'direction':  ['N','N','N','N'],
            'timestamp': [0,1,2,3],
            'flow':      [10.0, np.nan, np.nan, 40.0],
            'mask_flag': [False, True, True, False],
        })

    def test_linear_interpolation(self):
        df = self.base.copy()
        out = temporal_fill(df, 'flow', group_cols=['station_id','direction'], time_col='timestamp')
        # mask_flag=True 的两点应被线性填充：位置1→20, 位置2→30
        np.testing.assert_allclose(out['flow'].values, [10,20,30,40])

    def test_single_valid(self):
        # 只有一个有效点时，不做插值
        df = self.base.copy()
        df.loc[[2,3], 'mask_flag'] = True
        df.loc[[2,3], 'flow'] = np.nan
        out = temporal_fill(df, 'flow')
        # 只有 idx=0 有效，其它保留原 NaN
        self.assertEqual(out.loc[0,'flow'], 10.0)
        self.assertTrue(np.isnan(out.loc[1,'flow']))
        self.assertTrue(np.isnan(out.loc[2,'flow']))
        self.assertTrue(np.isnan(out.loc[3,'flow']))

    def test_no_mask(self):
        # 没有 mask_flag → flow 不变
        df = self.base.copy()
        df['mask_flag'] = False
        df.loc[1,'flow'] = 20
        df.loc[2,'flow'] = 30
        out = temporal_fill(df, 'flow')
        np.testing.assert_array_equal(out['flow'], df['flow'])

if __name__ == '__main__':
    unittest.main()
