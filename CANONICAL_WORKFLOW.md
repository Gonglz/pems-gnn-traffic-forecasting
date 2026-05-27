# PeMS Traffic Forecasting Workflow

This document records the maintained workflow for the PeMS traffic forecasting
system. The goal is reproducible data preparation, graph alignment, model
training, and inference on real traffic sensor data. Run smoke or benchmark
outputs under `/scratch2` and keep production arrays/checkpoints untouched
unless intentionally rebuilding them.

## Environment

Use the preconfigured environment:

```bash
/scratch/lgong1/envs/traffic-env/bin/python
```

For DDP smoke or benchmark runs, select the target GPUs explicitly:

```bash
CUDA_VISIBLE_DEVICES=2,3
```

## Data and Feature Pipeline

Existing upstream artifacts:

- `pems_data/step31_fillExter.parquet`
- `pems_data/step34_maskMix.parquet`
- `pems_data/step01_d07_meta.csv`
- `pems_data/weather_5min_history.parquet`

Safe interpolation:

```bash
python data_process/step50_Fill.py \
  --base-dir /scratch/lgong1/finalproject/pems_data

python data_process/step51_refill.py \
  --raw /scratch/lgong1/finalproject/pems_data/step31_fillExter.parquet \
  --mask /scratch/lgong1/finalproject/pems_data/step34_maskMix.parquet \
  --interp /scratch/lgong1/finalproject/pems_data/step50_interpolated_fastest.parquet \
  --output /scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet
```

Fill validation:

```bash
python data_process/step50_fillCheck.py \
  --raw /scratch/lgong1/finalproject/pems_data/step31_fillExter.parquet \
  --mask /scratch/lgong1/finalproject/pems_data/step34_maskMix.parquet \
  --interp /scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet
```

Topology and neighborhoods:

```bash
python data_process/step52_buildTopo.py \
  --base-dir /scratch/lgong1/finalproject/pems_data

python data_process/step62_precompute_neighbors.py \
  --base-dir /scratch/lgong1/finalproject/pems_data
```

Training arrays with station-aligned weather:

```bash
python pems_data/step70_make_xy_weather.py \
  --base-dir /scratch/lgong1/finalproject/pems_data \
  --graph-nodes-pkl /scratch/lgong1/finalproject/pems_data/step52_graph_nodes.pkl
```

Expected alignment: `X_ext.npy`, `Y.npy`, `sids.npy`,
`timestamps.npy`, `step52_edge_index.pt`, and `step62_neighbors.pkl` all refer
to the same graph node order.

## Training and Inference

Canonical training entry point:

```bash
python -m model.train_sampler \
  --data-dir /scratch/lgong1/finalproject/pems_data \
  --edge-index-path /scratch/lgong1/finalproject/pems_data/step52_edge_index.pt \
  --checkpoint-dir /scratch2/lgong1/finalproject_runs/<run-id>
```

DDP smoke or benchmark entry point:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  -m model.train_sampler \
  --data-dir /scratch2/lgong1/finalproject_smoke/<timestamp>/pems_data \
  --edge-index-path /scratch2/lgong1/finalproject_smoke/<timestamp>/pems_data/step52_edge_index.pt \
  --checkpoint-dir /scratch2/lgong1/finalproject_smoke/<timestamp>/checkpoints_ddp \
  --epochs 1 --max-train-times 2 --max-val-times 1 --batch-size 2 --num-workers 0 --no-amp
```

Inference accepts either a real PeMS station ID or an internal graph node index:

```bash
python -m model.predict_and_plot \
  --data-dir /scratch/lgong1/finalproject/pems_data \
  --edge-index-path /scratch/lgong1/finalproject/pems_data/step52_edge_index.pt \
  --model-path /scratch/lgong1/finalproject/best_rf_gnn_dynamic_sampler.pth \
  --station-id <real-pems-station-id> \
  --time-idx 0
```

## Evidence-First Next Steps

Before full retraining, run a medium-scale benchmark that reports:

- MAE/RMSE/MAPE against simple baselines.
- Training time and memory for GraphSAGE + weather.
- 1 GPU vs 2 GPU DDP epoch time and speedup.
- Visualization artifacts: topology, prediction-vs-truth, and error heatmap.

Detailed evidence plan:

- `/scratch/lgong1/finalproject/EVIDENCE_PLAN.md`

## Historical Paths

The following files are kept for reference but are not the current workflow
entry points:

- `model/train_final.py`
- `model/train_faster.py`
- `data_process/step40_*`
- custom SpMM experiments in `model/spmm_ext.py`

The custom SpMM path is not the current optimization priority. The stronger
story for this project is robust PeMS data preparation, graph-node alignment,
GraphSAGE/NeighborSampler modeling, and distributed training correctness.

