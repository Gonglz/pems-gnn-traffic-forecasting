# PeMS Spatiotemporal Traffic Forecasting with Graph Neural Networks

End-to-end traffic forecasting pipeline on real PeMS sensor data, covering raw data cleaning, anomaly-aware imputation, graph construction, station-aligned weather features, PyTorch Geometric GraphSAGE training, multi-GPU DDP scaling, inference samples, and reproducible Evidence Pack reporting.

## Highlights

- Built a full data-to-model workflow for PeMS traffic sensors: preprocessing, anomaly masking, imputation, sensor graph construction, weather alignment, train/eval/inference, and result archiving.
- Implemented GraphSAGE training with PyG `NeighborSampler`, AMP, TF32, and 2-GPU DDP using fair flat-batch sharding over `(time_idx, seed-node-batch)` workloads.
- Added model-quality improvements: train-only station-wise normalization, temporal encodings, station embeddings, horizon-specific normalized loss, and an MLP/no-graph baseline.
- Ran a full-scale GraphSAGE P2 training proof over `33177` timestamps, `4883` graph-aligned stations, and `162,003,291` timestamp-station pairs.

## Evidence Summary

The main results are documented in:

- `EVIDENCE_RESULTS.md`
- `PROJECT_REPORT.md`
- `model/模型训练流程与加速方法.md`
- `data_process/数据预处理流程与加速方法.md`

Key evidence runs:

- P2 4-week GraphSAGE: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-graphsage-20260525-125437`
- P2 4-week MLP/no-graph: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-mlp-20260525-141756`
- Full-scale GraphSAGE 10 epoch: `/scratch2/lgong1/finalproject_gpu_parallel_test/full-p2-graphsage-10ep-20260525-152216`

Full-scale all-horizon raw-flow metrics:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Last value | 62.8537 | 116.6103 | 55.9902 |
| Historical average | 77.4977 | 106.8358 | 53.9472 |
| GraphSAGE P2 full-scale, 10 epochs | 67.9174 | 97.6607 | 74.2615 |

The full-scale run is strong train/eval/archive evidence and shows RMSE reduction versus both baselines. It should not be described as a blanket win over every baseline and metric.

## Repository Scope

Large PeMS raw/intermediate arrays, parquet files, checkpoints, and generated experiment outputs are intentionally excluded from GitHub. The repository tracks source code and reporting artifacts only.

The local data directory used for experiments was:

```text
/scratch/lgong1/finalproject/pems_data
```

The local Evidence Pack output root was:

```text
/scratch2/lgong1/finalproject_gpu_parallel_test
```

## Main Entry Points

Prepare or run an Evidence Pack:

```bash
/scratch/lgong1/envs/traffic-env/bin/python -m model.evidence_pack run \
  --source-data-dir /scratch/lgong1/finalproject/pems_data \
  --output-root /scratch2/lgong1/finalproject_gpu_parallel_test/<run-id> \
  --epochs 10 \
  --batch-size 1024 \
  --num-workers 4 \
  --skip-single \
  --skip-rf \
  --ddp-gpus 3,4 \
  --model-type graphsage \
  --normalization station-wise-flow \
  --station-embedding-dim 16 \
  --ddp-shard-mode flat-batch
```

Run focused tests:

```bash
/scratch/lgong1/envs/traffic-env/bin/python -m unittest model/test_worker_c.py model/test_evidence_pack.py
```

## Notes

This project is best framed as a real-data ML engineering and spatiotemporal GNN pipeline. Historical custom CUDA/SpMM experiments remain in the codebase, but the current Evidence Pack mainline uses PyTorch Geometric GraphSAGE with NeighborSampler, AMP, TF32, and DDP.
