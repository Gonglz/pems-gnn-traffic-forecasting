#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Multi‑Head RF‑GraphSAGE 训练脚本
- 三尺度物理感受野：5/15/30min edge_index
- 特征：X_ext.npy (flow, occupancy, speed, tavg, pcpn, is_weekend)
- 标签：Y.npy (flow)
- 支持混合精度、梯度累积、torch.compile、DDP 五卡并行 & 本地 debug
/scratch/lgong1/envs/traffic-env/bin/python /scratch/lgong1/finalproject/model/train.py
[Local] debug=False, DEVICE=cuda:0
Dataset len=33171, feature dim=6
Epoch 1/20:   0%|          | 0/33171 [00:00<?, ?it/s][2025-05-06 21:34:57,547] torch._inductor.utils: [WARNING] using triton random, expect difference from eager
[2025-05-06 21:34:59,840] torch._inductor.utils: [WARNING] using triton random, expect difference from eager
[2025-05-06 21:35:00,418] torch._inductor.utils: [WARNING] using triton random, expect difference from eager
Epoch 1/20: 100%|██████████| 33171/33171 [07:15<00:00, 76.09it/s, loss=2.67e+4]
Epoch 2/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 1 done, avg loss=26714.262086
Epoch 2/20: 100%|██████████| 33171/33171 [07:10<00:00, 77.12it/s, loss=2.55e+4]
Epoch 3/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 2 done, avg loss=25465.810419
Epoch 3/20: 100%|██████████| 33171/33171 [07:59<00:00, 69.13it/s, loss=2.54e+4]
Epoch 3 done, avg loss=25371.504094
Epoch 4/20: 100%|██████████| 33171/33171 [08:23<00:00, 65.94it/s, loss=2.53e+4]
Epoch 4 done, avg loss=25264.830338
Epoch 5/20: 100%|██████████| 33171/33171 [08:18<00:00, 66.52it/s, loss=2.52e+4]
Epoch 6/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 5 done, avg loss=25200.214391
Epoch 6/20: 100%|██████████| 33171/33171 [08:28<00:00, 65.25it/s, loss=2.51e+4]
Epoch 6 done, avg loss=25137.695842
Epoch 7/20: 100%|██████████| 33171/33171 [08:13<00:00, 67.20it/s, loss=2.51e+4]
Epoch 7 done, avg loss=25086.290911
Epoch 8/20: 100%|██████████| 33171/33171 [07:39<00:00, 72.17it/s, loss=2.5e+4]
Epoch 8 done, avg loss=25045.428649
Epoch 9/20: 100%|██████████| 33171/33171 [07:28<00:00, 73.89it/s, loss=2.5e+4]
Epoch 10/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 9 done, avg loss=24999.940910
Epoch 10/20: 100%|██████████| 33171/33171 [07:05<00:00, 77.99it/s, loss=2.5e+4]
Epoch 11/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 10 done, avg loss=24963.500285
Epoch 11/20: 100%|██████████| 33171/33171 [07:05<00:00, 77.90it/s, loss=2.49e+4]
Epoch 11 done, avg loss=24922.441081
Epoch 12/20: 100%|██████████| 33171/33171 [07:17<00:00, 75.87it/s, loss=2.49e+4]
Epoch 13/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 12 done, avg loss=24876.887545
Epoch 13/20: 100%|██████████| 33171/33171 [07:07<00:00, 77.54it/s, loss=2.48e+4]
Epoch 14/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 13 done, avg loss=24804.501604
Epoch 14/20: 100%|██████████| 33171/33171 [07:15<00:00, 76.15it/s, loss=2.48e+4]
Epoch 14 done, avg loss=24761.691879
Epoch 15/20: 100%|██████████| 33171/33171 [06:43<00:00, 82.11it/s, loss=2.47e+4]
Epoch 15 done, avg loss=24724.821031
Epoch 16/20: 100%|██████████| 33171/33171 [06:35<00:00, 83.96it/s, loss=2.47e+4]
Epoch 17/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 16 done, avg loss=24708.506112
Epoch 17/20: 100%|██████████| 33171/33171 [05:56<00:00, 93.15it/s, loss=2.47e+4]
Epoch 17 done, avg loss=24674.861594
Epoch 18/20: 100%|██████████| 33171/33171 [06:39<00:00, 82.97it/s, loss=2.46e+4]
Epoch 18 done, avg loss=24641.077984
Epoch 19/20: 100%|██████████| 33171/33171 [06:27<00:00, 85.64it/s, loss=2.46e+4]
Epoch 20/20:   0%|          | 0/33171 [00:00<?, ?it/s]Epoch 19 done, avg loss=24596.050300
Epoch 20/20: 100%|██████████| 33171/33171 [06:50<00:00, 80.82it/s, loss=2.46e+4]
Epoch 20 done, avg loss=24555.965422

进程已结束，退出代码为 0
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
import pdb

# ─── 超参数 ────────────────────────────────────────────────────────────────
DATA_DIR     = '/scratch/lgong1/finalproject/pems_data'
EPOCHS       = 20
LR           = 1e-3
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 1      # 每进程每 batch 用一个全图
ACCUM_STEPS  = 4      # 梯度累积步数
NUM_WORKERS  = 4
USE_AMP      = True   # 混合精度
USE_COMPILE  = True   # torch.compile
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='调试模式：使用 DataParallel 并触发 pdb')
    p.add_argument('--local_rank', type=int,
                   default=int(os.getenv('LOCAL_RANK', 0)))
    return p.parse_args()

args = parse_args()

# ─── DDP or 本地 ───────────────────────────────────────────────────────────
use_ddp = (not args.debug) and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
if use_ddp:
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    DEVICE = torch.device(f'cuda:{args.local_rank}')
    rank = dist.get_rank(); world_size = dist.get_world_size()
    print(f"[DDP] rank={rank}/{world_size}, local_rank={args.local_rank}")
else:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Local] debug={args.debug}, DEVICE={DEVICE}")

# ─── Dataset ────────────────────────────────────────────────────────────────
class RFGraphDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        # 特征和标签
        X = np.load(os.path.join(root, 'X_ext.npy'))  # (T, N, F)
        Y = np.load(os.path.join(root, 'Y.npy'))      # (T, N)
        self.T, self.N, self.F = X.shape
        # 三个时间尺度的 delta
        self.delta5, self.delta15, self.delta30 = 1, 3, 6
        self.max_delta = max(self.delta5, self.delta15, self.delta30)
        # 转成 tensor
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        # 读 edge_index
        self.edge5  = torch.load(os.path.join(root, 'edge_index_5min.pt'))
        self.edge15 = torch.load(os.path.join(root, 'edge_index_15min.pt'))
        self.edge30 = torch.load(os.path.join(root, 'edge_index_30min.pt'))
        # 构造 Data 列表
        self.data_list = []
        for t in range(self.T - self.max_delta):
            data = Data(
                x = self.X[t],                           # [N, F]
                edge_index5  = self.edge5,
                edge_index15 = self.edge15,
                edge_index30 = self.edge30,
                y5  = self.Y[t + self.delta5 ].unsqueeze(-1),
                y15 = self.Y[t + self.delta15].unsqueeze(-1),
                y30 = self.Y[t + self.delta30].unsqueeze(-1),
            )
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# ─── Model ─────────────────────────────────────────────────────────────────
class MultiHeadRFGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = [SAGEConv(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs   = nn.ModuleList(layers)
        self.dropout = dropout
        # 三个预测头
        self.head5  = nn.Linear(hidden_dim, 1)
        self.head15 = nn.Linear(hidden_dim, 1)
        self.head30 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x
        def run(ei):
            h = x
            for conv in self.convs:
                h = conv(h, ei)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return h
        h5  = run(data.edge_index5)
        h15 = run(data.edge_index15)
        h30 = run(data.edge_index30)
        return self.head5(h5), self.head15(h15), self.head30(h30)

# ─── 训练函数 ─────────────────────────────────────────────────────────────
def train():
    # 数据集 & sampler & loader
    dataset = RFGraphDataset(DATA_DIR)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    # Debug 下打印首样本并断点
    print(f"Dataset len={len(dataset)}, feature dim={dataset.F}")
    if args.debug:
        print("First sample:", dataset.get(0))
        print("Entering pdb...")
        pdb.set_trace()

    # 模型、并行封装、compile
    model = MultiHeadRFGraphSAGE(dataset.F, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    elif args.debug and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using DataParallel")

    if USE_COMPILE:
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=USE_AMP)

    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.allow_tf32    = True

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        total_loss = 0.0

        loop = tqdm(loader,
                    desc=f"Epoch {epoch}/{EPOCHS}",
                    disable=(use_ddp and args.local_rank != 0))
        for step, data in enumerate(loop):
            data = data.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                y5, y15, y30 = model(data)
                loss = (F.mse_loss(y5, data.y5) +
                        F.mse_loss(y15, data.y15) +
                        F.mse_loss(y30, data.y30)) / ACCUM_STEPS
            scaler.scale(loss).backward()
            # 梯度累积
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * ACCUM_STEPS
            if args.debug and step >= 2:
                print("Debug batch done, exiting.")
                return
            loop.set_postfix(loss=total_loss / ((step + 1) * BATCH_SIZE))

        if not (use_ddp and args.local_rank != 0):
            print(f"Epoch {epoch} done, avg loss={total_loss/len(dataset):.6f}")

    # 保存模型（只有主进程）
    if not (use_ddp and args.local_rank != 0):
        torch.save(model.state_dict(),
                   os.path.join(DATA_DIR, 'rf_graphsage.pth'))

if __name__ == '__main__':
    train()
