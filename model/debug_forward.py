#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/debug_forward.py

本脚本自动把项目根目录加入 sys.path，支持绝对导入：
  from model.dataset_full import RFGraphDatasetFull
  from model.gnn_full     import MultiHeadRFGraphSAGEDyn
"""

import os
import sys

# ─── 把项目根目录（即包含 model/ 的上一级）加到 sys.path ────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 现在可以用绝对导入了
from dataset_full import RFGraphDatasetFull
from gnn_full     import MultiHeadRFGraphSAGEDyn

import torch
#!/usr/bin/env python3
import os, sys
# 把项目根目录插入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 绝对导入
from dataset_full import RFGraphDatasetFull
from gnn_full     import MultiHeadRFGraphSAGEDyn

import torch

def main():
    ds = RFGraphDatasetFull()
    data = ds.get(0).to('cuda')
    model = MultiHeadRFGraphSAGEDyn(ds.F,64,2,0.3).to('cuda').eval()
    with torch.no_grad():
        y5,y15,y30 = model(data)
    print("OK:", y5.shape, y15.shape, y30.shape)

if __name__=='__main__':
    main()
def main():
    print("Loading dataset…")
    ds = RFGraphDatasetFull()
    print(f"T={ds.T}, N={ds.N}, F={ds.F}")

    print("Fetching sample #0…")
    data = ds.get(0).to('cuda')
    print("  x.shape   =", data.x.shape)
    print("  nbr5.shape=", data.nbr5.shape)
    print("  y5.shape  =", data.y5.shape)

    print("Instantiating model…")
    model = MultiHeadRFGraphSAGEDyn(ds.F, hidden_dim=64, num_layers=2, dropout=0.3).to('cuda')
    model.eval()

    print("Running forward…")
    with torch.no_grad():
        y5, y15, y30 = model(data)
    print("  y5.shape  =", y5.shape)
    print("  y15.shape =", y15.shape)
    print("  y30.shape =", y30.shape)
    print("y5 sample:", y5[:5].cpu().numpy().flatten())

    print("✅ Forward pass successful!")

if __name__ == '__main__':
    main()
