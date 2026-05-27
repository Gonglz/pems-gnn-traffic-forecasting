#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/train_final.py

notetrainingnote - RF‑GraphSAGE note GNN with custom SpMM conv
- dynamicnote (RF)
- note SpMM kernel (GraphSAGEDynConvFinal)
- noterowsnote 5/15/30 min
- 80/20 Train/Val split
- AMP note + note + DDP note
- note (tqdm)
(traffic-env) lgong1@microway:/scratch/lgong1/finalproject$ export CUDA_VISIBLE_DEVICES=0,1,2,3,4
(traffic-env) lgong1@microway:/scratch/lgong1/finalproject$ torchrun --nproc_per_node=5 -m model.train_final
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
Using /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Using /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117/spmm_ext/build.ninja...
Using /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Using /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Building extension module spmm_ext...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module spmm_ext...
Loading extension module spmm_ext...
Loading extension module spmm_ext...
Using /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/CAMPUS/lgong1/.cache/torch_extensions/py310_cu117/spmm_ext/build.ninja...
Building extension module spmm_ext...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module spmm_ext...
Loading extension module spmm_ext...
[DDP] world_size=5, backend=nccl
Dataset: T=33177, N=4883, F=6, train=26536, val=6635

=== Epoch 1/20 ===
[Epoch 1] avg MSE loss = 23297.782334  14min
epoch2 8min
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
from tqdm import tqdm

# note model note
from.dataset_full import RFGraphDatasetFull
from.gnn_final     import MultiHeadRFGraphSAGEDyn

# ─── note ───────────────────────────────────────────────────────────
EPOCHS       = 20
LR           = 1e-3
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 1      # note GPU note step note
ACCUM_STEPS  = 4
NUM_WORKERS  = 4
USE_AMP      = True
DEBUG_STEPS  = 10     # debug>0 note DEBUG_STEPS note, note

# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true',
                   help='note DEBUG_STEPS note batch, note')
    return p.parse_args()

args = parse_args()

# ─── DDP / DEVICE note ─────────────────────────────────────────────────────
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
# ───────────────────────────────────────────────────────────────────────────

def train():
    # 1) notedataset & note
    ds = RFGraphDatasetFull()
    train_ds = ds.train_list
    val_ds   = ds.val_list
    if rank == 0:
        print(f"Dataset: T={ds.T}, N={ds.N}, F={ds.F}, train={len(train_ds)}, val={len(val_ds)}")

    # 2) DataLoader
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

    # 3) model & note & AMP
    model = MultiHeadRFGraphSAGEDyn(
        in_dim=ds.F,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    """    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])"""

    # --- Torch 2.0 build: note DDP(note debug)note ---

    if not use_ddp:
       model = torch.compile(model, fullgraph=True)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids = [local_rank],
            find_unused_parameters = True,  # <- noterows
            )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=USE_AMP)

    torch.backends.cudnn.benchmark  = True
    torch.backends.cudnn.allow_tf32 = True

    # 4) training＋note
    for epoch in range(1, EPOCHS+1):
        model.train()
        if use_ddp:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0:
            print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[Epoch {epoch} train]",
            ncols=100,
            leave=False,
        )
        for step, data in train_loop:
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

            if rank == 0:
                train_loop.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.1f}")

            if args.debug and step+1 >= DEBUG_STEPS:
                if rank == 0:
                    train_loop.write(f"[DEBUG] note {DEBUG_STEPS} note, notefirstnote train loop")
                break

        # 4.1 training Loss note
        div = min(DEBUG_STEPS, len(train_ds)) if (args.debug and DEBUG_STEPS>0) else len(train_ds)
        avg_loss = total_loss / div
        if rank == 0:
            print(f"[Epoch {epoch}] avg MSE loss = {avg_loss:.6f}")

        # 4.2 note
        model.eval()
        total_mse5 = total_mae5 = 0.0

        val_loop = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"[Epoch {epoch} val]  ",
            ncols=100,
            leave=False,
        )
        with torch.no_grad():
            for vstep, vdata in val_loop:
                if args.debug and vstep+1 >= DEBUG_STEPS:
                    if rank == 0:
                        val_loop.write(f"[DEBUG] note {DEBUG_STEPS} note, notefirstnote val loop")
                    break

                vdata = vdata.to(DEVICE)
                p5, _, _ = model(vdata)
                total_mse5 += F.mse_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()
                total_mae5 += F.l1_loss(p5, vdata.y5.to(DEVICE), reduction='sum').item()

                if rank == 0:
                    rmse5 = (total_mse5 / ((vstep+1)*ds.N))**0.5
                    val_loop.set_postfix(rmse5=f"{rmse5:.2f}")

        # note
        samples = (DEBUG_STEPS if (args.debug and DEBUG_STEPS>0) else len(val_ds)) * ds.N
        rmse5 = (total_mse5 / samples)**0.5
        mae5  = total_mae5 / samples
        if rank == 0:
            print(f"[VAL ] RMSE5={rmse5:.2f}, MAE5={mae5:.2f}")

    # 5) savemodel
    if rank == 0:
        torch.save(model.state_dict(), 'rf_gnn_dynamic_final.pth')
        print("Saved checkpoint: rf_gnn_dynamic_final.pth")

if __name__ == '__main__':
    train()
