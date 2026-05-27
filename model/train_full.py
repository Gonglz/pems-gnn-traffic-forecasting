#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full.py

Dynamic RF‑GraphSAGE multi‑head training with train/val split.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader

# absolute imports from the model package
from.dataset_full import RFGraphDatasetFull
from.gnn_full     import MultiHeadRFGraphSAGEDyn

# ─── Hyperparameters ────────────────────────────────────────────
EPOCHS       = 20
LR           = 1e-3
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 1      # full‑graph per batch per GPU
ACCUM_STEPS  = 4
NUM_WORKERS  = 4
USE_AMP      = True
DEBUG_STEPS  = 10     # only used in debug mode
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='single‑GPU quick debug (no DDP)')
    return p.parse_args()

args = parse_args()

# ─── DDP / DEVICE setup ─────────────────────────────────────────────
use_ddp = not args.debug and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
if use_ddp:
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"[DDP] world_size={world_size}, backend=nccl")
else:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rank, world_size = 0, 1
    print(f"[Local] debug={args.debug}, DEVICE={DEVICE}")
# ────────────────────────────────────────────────────────────────────

def train():
    # 1) load dataset & split
    ds = RFGraphDatasetFull()
    train_ds = ds.train_list
    val_ds   = ds.val_list
    if rank == 0:
        print(f"Dataset: T={ds.T}, N={ds.N}, F={ds.F}, train={len(train_ds)}, val={len(val_ds)}")

    # 2) dataloaders
    train_sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader  = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 3) model, optimizer, scaler
    model = MultiHeadRFGraphSAGEDyn(
        in_dim=ds.F,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif args.debug and torch.cuda.device_count() > 1:
        # skip DataParallel in debug, or comment out entirely
        pass

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=USE_AMP)

    # 4) training + validation loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        if use_ddp:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0:
            print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        for step, data in enumerate(train_loader):
            data = data.to(DEVICE)
            with autocast(enabled=USE_AMP):
                y5, y15, y30 = model(data)
                loss = (
                    F.mse_loss(y5,  data.y5.to(DEVICE)) +
                    F.mse_loss(y15, data.y15.to(DEVICE)) +
                    F.mse_loss(y30, data.y30.to(DEVICE))
                ) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS

            # debug short‑run
            if args.debug and step + 1 >= DEBUG_STEPS:
                if rank == 0:
                    print(f"[DEBUG] ran {DEBUG_STEPS} steps, breaking early")
                break

            # first‐batch debug print
            if step == 0 and rank == 0:
                print(f"[DEBUG] Step 0 loss = {loss.item()*ACCUM_STEPS:.4f}")

        """ # report train loss
        avg_loss = total_loss / len(train_ds if not args.debug else min(DEBUG_STEPS, len(train_ds)))
        if rank == 0:
            print(f"[Epoch {epoch}] avg MSE loss = {avg_loss:.6f}")"""
        # report train loss
        if args.debug:
                    divisor = min(DEBUG_STEPS, len(train_ds))
        else:
                   divisor = len(train_ds)
        avg_loss = total_loss / divisor

        if rank == 0:
                        print(f"[Epoch {epoch}] avg MSE loss = {avg_loss:.6f}")

        # - validation -
        model.eval()
        """total_mse5 = total_mae5 = 0.0
        with torch.no_grad():
            for vdata in val_loader:
                vdata = vdata.to(DEVICE)
                p5, _, _ = model(vdata)
                total_mse5 += F.mse_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()
                total_mae5 += F.l1_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()
        rmse5 = (total_mse5 / len(val_ds))**0.5
        mae5  = total_mae5 / len(val_ds)
        if rank == 0:
            print(f"[VAL ] RMSE5={rmse5:.2f}, MAE5={mae5:.2f}")"""
        total_mse5 = total_mae5 = 0.0
        with torch.no_grad():
            for vdata in val_loader:
                vdata = vdata.to(DEVICE)
                p5, _, _ = model(vdata)
                # sum over all nodes in this batch
                total_mse5 += F.mse_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()
                total_mae5 += F.l1_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()

        # note:  note x note
        num_val_windows = len(val_ds)
        num_nodes = ds.N  # 4883
        total_samples = num_val_windows * num_nodes

        rmse5 = (total_mse5 / total_samples) ** 0.5
        mae5 = total_mae5 / total_samples

        if rank == 0:
            print(f"[VAL ] RMSE5={rmse5:.2f}, MAE5={mae5:.2f}")

    # 5) save checkpoint
    if rank == 0:
        ckpt = 'rf_gnn_dynamic.pth'
        torch.save(model.state_dict(), ckpt)
        print(f"Model checkpoint saved to {ckpt}")

if __name__ == '__main__':
    train()
