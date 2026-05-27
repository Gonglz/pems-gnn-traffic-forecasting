#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/train_optimized.py

notetrainingnote - note train_faster note:
- note GPU note(note dataset_full note)
- Torch 2.0 fullgraph build(note/DEBUG)
- DDP note with find_unused_parameters
- AMP note + note
- CuDNN TF32 + Benchmark
- DataLoader note workers + note
- note & note
- tqdm note
=== Epoch 1/20 ===
[Epoch 1 train]:  76%|████████████████████▍      | 4024/5308 [10:02<03:12,  6.67it/s, loss=109846.3][Epoch 1 train]:  74%|███████████████████████████████           | 3925/5308 [09:
[Epoch 1] avg MSE loss = 119168.690396
[DEBUG] note 663 note, notefirstnote
[VAL ] RMSE5=160.69, MAE5=107.43
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# note
from.dataset_full import RFGraphDatasetFull
from.gnn_final     import MultiHeadRFGraphSAGEDyn

# ─── note ───────────────────────────────────────────────────────────
EPOCHS        = 20
LR            = 1e-3
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 1
ACCUM_STEPS   = 4
NUM_WORKERS   = 4
USE_AMP       = True
DEBUG_STEPS   = 10    # debug note
VAL_PERIOD    = 2     # note
VAL_FRACTION  = 0.1   # note, first VAL_FRACTION note
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='DEBUG note: note DEBUG_STEPS note')
    return p.parse_args()

args = parse_args()

# ─── DDP / DEVICE note ─────────────────────────────────────────────────
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
# ────────────────────────────────────────────────────────────────────────

def train():
    # 1. dataset
    ds = RFGraphDatasetFull()       # note neighbors GPU note
    train_ds = ds.train_list
    val_ds   = ds.val_list
    if rank == 0:
        print(f"Dataset: T={ds.T}, N={ds.N}, F={ds.F}, train={len(train_ds)}, val={len(val_ds)}")

    # 2. DataLoader
    train_sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # 3. model & note & AMP
    model = MultiHeadRFGraphSAGEDyn(
        in_dim=ds.F,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    # 3.1 Torch 2.0 build(note/DEBUG)
    if not use_ddp:
        model = torch.compile(model, fullgraph=True)

    # 3.2 DDP note
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=USE_AMP)

    # 3.3 CuDNN TF32 & Benchmark
    torch.backends.cudnn.benchmark  = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 4. training/note
    for epoch in range(1, EPOCHS+1):
        model.train()
        if use_ddp:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0:
            print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        # -- training --
        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[Epoch {epoch} train]",
            ncols=100, leave=False
        )
        for step, data in train_bar:
            data = data.to(DEVICE, non_blocking=True)
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
            if rank == 0:
                train_bar.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.1f}")

            if args.debug and step+1 >= DEBUG_STEPS:
                if rank == 0:
                    train_bar.write(f"[DEBUG] note {DEBUG_STEPS} notetrainingnote, notefirstnote")
                break

        # trainingnote Loss
        used_steps = min(DEBUG_STEPS, len(train_loader)) if (args.debug and DEBUG_STEPS>0) else len(train_loader)
        avg_loss = total_loss / used_steps
        if rank == 0:
            print(f"[Epoch {epoch}] avg MSE loss = {avg_loss:.6f}")

        # -- note --
        model.eval()
        total_mse5 = total_mae5 = 0.0

        full_val   = (epoch % VAL_PERIOD == 0) and not args.debug
        max_vsteps = DEBUG_STEPS if (args.debug and DEBUG_STEPS>0) else (
            len(val_loader) if full_val else max(1, int(len(val_loader)*VAL_FRACTION))
        )

        val_bar = tqdm(
            enumerate(val_loader),
            total=max_vsteps,
            desc=f"[Epoch {epoch} val]  ",
            ncols=100, leave=False
        )
        with torch.no_grad():
            for vstep, vdata in val_bar:
                if vstep+1 >= max_vsteps:
                    if rank == 0:
                        val_bar.write(f"[DEBUG] note {max_vsteps} note, notefirstnote")
                    break

                vdata = vdata.to(DEVICE, non_blocking=True)
                p5, _, _ = model(vdata)
                total_mse5 += F.mse_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()
                total_mae5 += F.l1_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()

                if rank == 0:
                    rmse5 = (total_mse5 / ((vstep+1)*ds.N))**0.5
                    val_bar.set_postfix(rmse5=f"{rmse5:.2f}")

        samples = max_vsteps * ds.N
        rmse5   = (total_mse5 / samples)**0.5
        mae5    = total_mae5 / samples
        if rank == 0:
            print(f"[VAL ] RMSE5={rmse5:.2f}, MAE5={mae5:.2f}")

    # 5. savemodel
    if rank == 0:
        torch.save(model.state_dict(), 'rf_gnn_dynamic_optimized.pth')
        print("Saved checkpoint: rf_gnn_dynamic_optimized.pth")


if __name__ == '__main__':
    train()
