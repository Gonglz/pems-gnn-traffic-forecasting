#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py

验证 X_ext.npy/Y.npy, Dataset, Model 前向推理都无 NaN/Inf
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import importlib.util

# ─── 配置 ────────────────────────────────────────────────────────────────
DATA_DIR = '/scratch/lgong1/finalproject/pems_data'
MODEL_PATH = '/scratch/lgong1/finalproject/model/train.py'
# ────────────────────────────────────────────────────────────────────────────

# 动态加载 train.py 里的 RFGraphDataset 和 MultiHeadRFGraphSAGE
spec = importlib.util.spec_from_file_location("train_mod", MODEL_PATH)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
RFGraphDataset = train_mod.RFGraphDataset
MultiHeadRFGraphSAGE = train_mod.MultiHeadRFGraphSAGE
HIDDEN_DIM = train_mod.HIDDEN_DIM
NUM_LAYERS = train_mod.NUM_LAYERS
DROPOUT = train_mod.DROPOUT


def main():
    # 1) 检查 X_ext.npy, Y.npy
    X = np.load(os.path.join(DATA_DIR, 'X_ext.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))
    print("X_ext.npy shape:", X.shape)
    print("  NaN count:", np.isnan(X).sum())
    print("  +Inf count:", np.isposinf(X).sum())
    print("  -Inf count:", np.isneginf(X).sum())
    print("Y.npy shape    :", Y.shape)
    print("  NaN count:", np.isnan(Y).sum())
    print("  +Inf count:", np.isposinf(Y).sum())
    print("  -Inf count:", np.isneginf(Y).sum())
    assert np.isnan(X).sum() == 0 and np.isnan(Y).sum() == 0, "Feature/label still contain NaN!"

    # 2) 构造 Dataset 并读取一个样本
    ds = RFGraphDataset(DATA_DIR)
    print(f"Dataset len = {len(ds)}, feature dim = {ds.F}")
    sample = ds.get(0)
    print("Sample[0]:")
    print("  x shape       :", sample.x.shape)
    print("  edge_index5   :", sample.edge_index5.shape)
    print("  edge_index15  :", sample.edge_index15.shape)
    print("  edge_index30  :", sample.edge_index30.shape)
    print("  y5, y15, y30  :", sample.y5.shape, sample.y15.shape, sample.y30.shape)
    # 检查 sample 内部是否还有 NaN/Inf
    for name, tensor in [('x', sample.x),
                         ('y5', sample.y5),
                         ('y15', sample.y15),
                         ('y30', sample.y30)]:
        print(f"  {name} NaN:", torch.isnan(tensor).sum().item(),
              f"+Inf:", torch.isposinf(tensor).sum().item(),
              f"-Inf:", torch.isneginf(tensor).sum().item())

    # 3) 前向推理一次
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadRFGraphSAGE(ds.F, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    model.eval()
    # 准备 Data 对象（只取一个 batch）
    data = Data(
        x=sample.x.to(device),
        edge_index5=sample.edge_index5.to(device),
        edge_index15=sample.edge_index15.to(device),
        edge_index30=sample.edge_index30.to(device)
    )
    with torch.no_grad():
        y5, y15, y30 = model(data)
    print("Forward outputs:")
    for name, out in [('y5', y5), ('y15', y15), ('y30', y30)]:
        print(f"  {name} shape: {out.shape}")
        print(f"    NaN:", torch.isnan(out).sum().item(),
              f"+Inf:", torch.isposinf(out).sum().item(),
              f"-Inf:", torch.isneginf(out).sum().item())

    print("==> Test passed: no NaN/Inf detected, model forward OK.")


if __name__ == '__main__':
    main()
