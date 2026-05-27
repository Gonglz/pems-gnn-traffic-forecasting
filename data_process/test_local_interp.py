import os, time, logging, sys
import numpy as np
import cudf, cupy as cp, dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.neighbors import NearestNeighbors
from numba import cuda, float32

# --- Configuration ---
BASE_DIR     = '/scratch/lgong1/finalproject/pems_data'
RAW_PARQ     = os.path.join(BASE_DIR, 'step31_fillExter.parquet')
MASK_PARQ    = os.path.join(BASE_DIR, 'step34_maskMix.parquet')
STATIONS_CSV = os.path.join(BASE_DIR, 'step01_d07_meta.csv')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'step40_interpolated_fastest.parquet')
THREADS      = 256
FEATURES     = ['flow','occupancy','speed']
K_NEIGHBORS  = 8
WORKERS      = 5
GPUS         = '0,1,2,3,4'
import numpy as np
import pandas as pd
from numba import cuda, float32

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numba import cuda, float32

# 要测试的 CUDA kernel
@cuda.jit
def local_kernel(feat, nbr, mask, out, K):
    i = cuda.grid(1)
    if i < feat.size:
        if mask[i]:
            acc = float32(0.0)
            cnt = 0
            for j in range(K):
                nb = nbr[i, j]
                if 0 <= nb < feat.size:
                    acc += feat[nb]
                    cnt += 1
            if cnt > 0:
                out[i] = acc / cnt
            else:
                out[i] = feat[i]
        else:
            out[i] = feat[i]

class TestLocalInterp(unittest.TestCase):
    def test_simple_interp(self):
        # 准备测试数据
        flow = np.array([10., 20., 30., 40.], dtype=np.float32)
        mask = np.array([False, True, False, False], dtype=np.bool_)
        nbr = np.array([
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3]
        ], dtype=np.int32)
        K = 2

        # 拷贝到 GPU
        d_flow = cuda.to_device(flow)
        d_mask = cuda.to_device(mask)
        d_nbr  = cuda.to_device(nbr)
        d_out  = cuda.device_array_like(d_flow)

        # 调用 kernel
        threads = 128
        blocks  = (flow.size + threads - 1) // threads
        local_kernel[blocks, threads](d_flow, d_nbr, d_mask, d_out, K)
        cuda.synchronize()

        # 拷回结果
        result = d_out.copy_to_host()

        # 期望值：只有第二个位置需要插值 (10+30)/2=20
        expected = np.array([10., 20., 30., 40.], dtype=np.float32)
        self.assertTrue(
            np.allclose(result, expected),
            msg=f"插值结果不正确: got {result}, expected {expected}"
        )

if __name__ == '__main__':
    unittest.main()
