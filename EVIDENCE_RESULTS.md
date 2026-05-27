# PeMS Traffic Forecasting Evidence Results

This file records the P0 medium-scale Evidence Pack and P1 extension runs for the PeMS spatiotemporal traffic forecasting system.

## P0 Evidence Pack

Run directory:

`/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936`

Code reference:

`/scratch2/lgong1/finalproject_backups/20260525-060748-evidence-code-preupload.tar.gz`

Post-check status:

- `config/`, `metrics/`, `plots/`, `logs/`, and both checkpoint directories were generated.
- Log scan for `nan|inf|traceback|error|cuda out of memory|killed` returned no matches.
- RF/MLP and true ablation experiments were intentionally skipped for this P0 run.

## Data Slice

- Time range: `2025-01-01 00:00:00` to `2025-01-07 23:55:00`
- Time steps: `2016` 5-minute timestamps
- Stations: `4883` graph-aligned PeMS stations from `step62_neighbors.pkl:graph_nodes`
- Features: `flow_interp`, `occupancy_interp`, `speed_interp`, `tavg`, `pcpn`, `is_weekend`
- Target: `flow_interp`
- Forecast horizons: `t+5`, `t+15`, and `t+30` minutes
- Metric split: held-out final 15% of valid time indices; baselines and checkpoint evaluation share the same split
- Array shapes: `X_ext.npy=(2016, 4883, 6)`, `Y.npy=(2016, 4883)`

## Metrics

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 23.2402 | 50.8365 | 39.7728 | naive temporal baseline, all horizons |
| Historical average | 21.6623 | 39.6356 | 37.7809 | train-split time-of-day baseline, all horizons |
| GraphSAGE + weather, 1 GPU | 27.9084 | 48.1118 | 50.6426 | 3 epochs, checkpoint from single GPU run |
| GraphSAGE + weather, 2 GPU DDP | 26.9821 | 48.0007 | 49.0806 | 3 epochs, checkpoint from 2 GPU DDP run |

Interpretation: this P0 run is valid as a reproducibility and scaling evidence pack, but the 3-epoch GraphSAGE checkpoints do not yet beat the historical-average baseline on MAE/MAPE. Do not claim accuracy improvement from this run; use it as the first closed-loop benchmark and tune/ablate in the next round.

## Scaling

| Setup | Epoch Time Sec | Speedup | Peak GPU Memory MB | Total Train Time Sec |
|---|---:|---:|---:|---:|
| 1 GPU | 274.4828 | 1.0000x | 23.8276 | 1019.3152 |
| 2 GPU DDP | 176.4142 | 1.5559x | 24.0918 | 658.1723 |

The DDP run processed non-overlapping time/node shards and produced a measurable training speedup for this medium slice.

## Plots

- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/topology.png`
- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/prediction_vs_truth_station_715898.png`
- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/error_heatmap.png`

## Deferred Evidence

The following rows in `metrics/ablation_metrics.csv` are placeholders with `notes=not_available` and should not be used as experimental conclusions yet:

- `without_weather`
- `without_graph_neighbors`
- `simple_ffill`

Historical note: this was true for P0. P1 later added real `without_weather` / `without_graph_neighbors` ablations, and P2 added the MLP/no-graph baseline.

## P1 Medium Tuning, Ablation, and Fair Scaling

Code changes in P1 added fair `flat-batch` DDP sharding over flattened `(time_idx, seed_node_batch)` work items, real `without_weather` and `without_graph_neighbors` ablations, `--skip-single` DDP-only Evidence Packs, and station/timestamp prediction CSV output.

Validation:

- Unit tests cover flat-batch DDP shard coverage, train-split weather replacement, self-loop graph mode, and skip-single scaling rows.
- Tiny Evidence Pack smoke passed with single GPU + 2 GPU DDP, `without_weather`, `self_loop`, checkpoint generation, model metrics, plots, and prediction CSV.
- Log scans for P1 runs returned no matches for `nan|inf|traceback|error|cuda out of memory|killed`.

### Fair Medium Scaling

Run directory:

`/scratch2/lgong1/finalproject_gpu_parallel_test/p1-fair-scaling-20260525-112038`

Same 1-week slice and medium workload as P0, but using fair `flat-batch` sharding. DDP rank work was balanced at `1,250,048` train time-node pairs per rank.

| Setup | Epoch Time Sec | Speedup | Peak GPU Memory MB | Notes |
|---|---:|---:|---:|---|
| 1 GPU | 18.7796 | 1.0000x | 23.1392 | baseline |
| 2 GPU DDP | 13.0925 | 1.4344x | 23.2461 | `ddp_shard_mode=flat-batch` |

Resume-safe wording: measured `1.43x` fair 2-GPU DDP speedup on the medium 1-week slice.

### Medium Tuned Run

Run directory:

`/scratch2/lgong1/finalproject_gpu_parallel_test/p1-medium-tune-20260525-112303`

Configuration: 10 epochs, 2 GPU DDP only, `batch_size=1024`, `max_train_times=512`, `max_val_times=128`, `dropout=0.1`, full held-out test split.

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 23.2402 | 50.8365 | 39.7728 | same held-out final 15% |
| Historical average | 21.6623 | 39.6356 | 37.7809 | train-split time-of-day baseline |
| GraphSAGE + weather, 2 GPU DDP tuned | 26.3360 | 47.6633 | 49.3620 | 10 epochs, fair flat-batch DDP |

Interpretation: tuning improved the DDP GraphSAGE MAE versus P0 (`26.9821 -> 26.3360`), but it still does not beat the historical-average baseline. Do not make an accuracy-improvement claim.

### Real Ablations

Each ablation used the same 1-week slice, held-out final 15%, 3 epochs, 2 GPU DDP, and fair `flat-batch` sharding.

| Ablation | Run Directory | MAE | RMSE | MAPE | Notes |
|---|---|---:|---:|---:|---|
| `without_weather` | `/scratch2/lgong1/finalproject_gpu_parallel_test/p1-ablation-without-weather-20260525-112650` | 26.8646 | 47.9663 | 48.8778 | `tavg/pcpn` replaced with train-split mean `[12.5958, 0.0]` |
| `without_graph_neighbors` | `/scratch2/lgong1/finalproject_gpu_parallel_test/p1-ablation-without-graph-20260525-112840` | 27.3594 | 48.1248 | 49.0564 | self-loop-only `edge_index`, `4883` edges |

Interpretation: these are now real ablation rows, not placeholders. In this short 3-epoch setting, both ablations are worse than the tuned full model, but none of these rows support a baseline-beating accuracy claim. `simple_ffill` remains deferred.

## P1 Full-Scale 2-GPU 1 Epoch Proof

Run directory:

`/scratch2/lgong1/finalproject_gpu_parallel_test/fullproof-20260525-113032`

This is not an accuracy-improvement run. It proves that the current Evidence Pack path can train, evaluate, plot, checkpoint, and archive a full-scale graph-aligned PeMS slice without rerunning raw preprocessing.

Data slice:

- Time steps: `33177` 5-minute timestamps
- Stations: `4883` graph-aligned PeMS stations
- Timestamp-station pairs: `162,003,291`
- Arrays: `X_ext.npy=(33177, 4883, 6)`, `Y.npy=(33177, 4883)`
- Target: `flow_interp`

DDP workload:

- Train time-node pairs: `129,575,288`
- Rank 0 train pairs: `64,787,644`
- Rank 1 train pairs: `64,787,644`
- Train steps per epoch: `132,680` global, `66,340` per rank
- GPUs used for final successful run: `3,4` because GPU `2` was later occupied by an external VLLM process

Artifacts:

- Checkpoints: `checkpoints/ddp_2gpu/best_rf_gnn_dynamic_sampler.pth`, `checkpoints/ddp_2gpu/rf_gnn_dynamic_sampler_last.pth`
- Metrics: `metrics/baseline_metrics.csv`, `metrics/model_metrics.csv`, `metrics/scaling_metrics.csv`, `metrics/ablation_metrics.csv`
- Plots: `plots/topology.png`, `plots/prediction_vs_truth_station_715898.png`, `plots/error_heatmap.png`
- Prediction CSV: `predictions/prediction_samples_station_715898.csv`

Full-scale proof metrics:

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 62.8537 | 116.6103 | 55.9902 | full held-out test split |
| Historical average | 77.4977 | 106.8358 | 53.9472 | full held-out test split |
| GraphSAGE + weather, 2 GPU DDP | 96.6328 | 125.7400 | 61.1638 | 1 epoch, model eval capped at 512 test timestamps |

Scaling/proof timing:

| Setup | Epoch Time Sec | Total Train Time Sec | Notes |
|---|---:|---:|---|
| 1 GPU | | | skipped by config |
| 2 GPU DDP | 572.7665 | 631.9582 | `ddp_shard_mode=flat-batch` |

Interpretation: the full-scale proof establishes large-scale train/eval/archive capability over `162M` timestamp-station pairs. It should be described as scale and systems evidence, not as a tuned accuracy result.

## P2 4-Week Model-Quality Runs

P2 kept the existing Evidence Pack infrastructure and focused only on model quality: target/window audit, train-only station-wise normalization, temporal encodings, trainable station embeddings, horizon-specific normalized loss, 30-epoch training with LR scheduling, and a required MLP/no-graph baseline.

Validation:

- Unit tests cover target leakage audit, station-wise train-only normalization and inverse transform, deterministic temporal features, MLP station embedding shape, existing DDP sharding, ablation, and scaling helpers.
- Tiny 2-GPU DDP smokes passed for both GraphSAGE and MLP with normalization stats, checkpoint generation, raw-unit model metrics, plots, and prediction CSV.
- P2 official log scans returned no matches for `nan|inf|traceback|error|cuda out of memory|killed`.

Data slice:

- Time range: `2025-01-01T00:00:00` to `2025-01-28T23:55:00`
- Time steps: `8064` 5-minute timestamps
- Stations: `4883` graph-aligned PeMS stations
- Timestamp-station pairs: `39,376,512`
- Valid source times: `8058`; train `5640`, val `1209`, test `1209`
- Full held-out test samples: `5,903,547` per horizon, `17,710,641` all horizons
- Target: `flow_interp`
- Horizons: `t+5`, `t+15`, `t+30` minutes

P2 configuration:

- `normalization=station-wise-flow`
- Input flow normalized from train source times only
- Each horizon target normalized from train target times only
- Eval and prediction inverse-transform outputs before MAE/RMSE/MAPE
- Added temporal features: `sin/cos time-of-day`, `sin/cos day-of-week`, `is_holiday`
- Preserved existing `is_weekend`
- Station embedding dim: `16`
- GPUs used: `3,4`
- Single-GPU run skipped by config; these runs compare model quality, not DDP speedup

Run directories:

- GraphSAGE: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-graphsage-20260525-125437`
- MLP/no-graph: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-mlp-20260525-141756`

All-horizon raw-flow metrics:

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 30.7889 | 62.6905 | 46.5187 | same 4-week held-out final 15% |
| Historical average | 41.5614 | 61.4426 | 64.2816 | train-split time-of-day baseline |
| MLP/no-graph + temporal + station embedding | 31.7255 | 49.4285 | 50.2572 | 30 epochs, no graph neighbors |
| GraphSAGE + temporal + station embedding | 30.7541 | 48.6064 | 47.2580 | 30 epochs, graph neighbors |

Per-horizon GraphSAGE metrics:

| Horizon | MAE | RMSE | MAPE | Samples |
|---:|---:|---:|---:|---:|
| 5 min | 30.3086 | 48.1538 | 46.8495 | 5,903,547 |
| 15 min | 31.1834 | 48.8846 | 47.6875 | 5,903,547 |
| 30 min | 30.7704 | 48.7777 | 47.2368 | 5,903,547 |

Training summary:

| Model | Best Val MSE | Epoch Time Sec | Total Train Sec | Peak GPU Memory MB |
|---|---:|---:|---:|---:|
| GraphSAGE | 0.8538 | 150.4224 | 4831.0005 | 25.7192 |
| MLP/no-graph | 0.8793 | 90.5520 | 2833.1541 | 20.8081 |

Memory values are recorded by the training script via PyTorch allocator stats and should not be used as a resume headline unless separately confirmed with `torch.cuda.max_memory_reserved()` and `nvidia-smi` logging.

Artifacts:

- GraphSAGE plots: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-graphsage-20260525-125437/plots/topology.png`, `prediction_vs_truth_station_715898.png`, `error_heatmap.png`
- GraphSAGE prediction CSV: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-graphsage-20260525-125437/predictions/prediction_samples_station_715898.csv`
- MLP plots: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-mlp-20260525-141756/plots/topology.png`, `prediction_vs_truth_station_715898.png`, `error_heatmap.png`
- MLP prediction CSV: `/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-mlp-20260525-141756/predictions/prediction_samples_station_715898.csv`

Interpretation:

- The target/window audit passed: `X[t]` is used as current timestamp input, and labels are `Y[t+1]`, `Y[t+3]`, `Y[t+6]`.
- Train-only station-wise normalization plus temporal features and station embeddings materially changed the result: the 4-week GraphSAGE run beats the historical-average baseline on MAE/RMSE/MAPE and beats the MLP/no-graph baseline on MAE/RMSE/MAPE.
- GraphSAGE only narrowly beats last-value on MAE and beats it clearly on RMSE, but it does not beat last-value on MAPE. Use precise wording; do not claim blanket improvement over every baseline and metric.
- The MLP/no-graph comparison supports a cautious graph-neighbor contribution claim for this P2 setting.

## P2 Full-Scale GraphSAGE 10 Epoch Run

Run directory:

`/scratch2/lgong1/finalproject_gpu_parallel_test/full-p2-graphsage-10ep-20260525-152216`

This run extends the P1 full-scale proof from 1 epoch to a 10-epoch full-scale GraphSAGE P2 training run. It uses the same P2 model-quality path as the 4-week runs: target/window audit, train-only station-wise flow normalization, temporal encodings, station embedding dim `16`, horizon-specific normalized loss, AMP, TF32, PyG NeighborSampler, and 2-GPU DDP flat-batch sharding.

Validation:

- Target/window audit passed with target `flow_interp`.
- Log scan returned no matches for `nan|inf|traceback|error|cuda out of memory|killed`.
- Best checkpoint, last checkpoint, normalization stats, epoch metrics, plots, prediction CSV, and full held-out metrics were generated.

Data slice:

- Time range: `2025-01-01T00:00:00` to `2025-04-26T23:55:00`
- Time steps: `33177` 5-minute timestamps
- Stations: `4883` graph-aligned PeMS stations
- Timestamp-station pairs: `162,003,291`
- Train pairs per epoch: `113,378,377`
- Full held-out test samples: `24,297,808` per horizon, `72,893,424` all horizons
- Target: `flow_interp`
- Horizons: `t+5`, `t+15`, `t+30` minutes

Training summary:

| Item | Value |
|---|---:|
| Epochs | 10 |
| Best val normalized MSE | 1.4332 |
| Final val normalized MSE | 1.4367 |
| Avg train epoch time | 632.0551 sec |
| Total train time | 6767.0505 sec |
| GPUs | `3,4` |
| Recorded peak GPU memory MB | 25.7173 |

The best validation checkpoint was saved at epoch 9. Validation MSE improved from `1.4720` at epoch 1 to `1.4332` at epoch 9.

Full held-out raw-flow metrics:

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 62.8537 | 116.6103 | 55.9902 | full held-out final 15% |
| Historical average | 77.4977 | 106.8358 | 53.9472 | train-split time-of-day baseline |
| GraphSAGE P2 full-scale, 10 epochs | 67.9174 | 97.6607 | 74.2615 | full held-out final 15%, no eval cap |

Per-horizon GraphSAGE metrics:

| Horizon | MAE | RMSE | MAPE | Samples |
|---:|---:|---:|---:|---:|
| 5 min | 66.8890 | 97.4173 | 73.9431 | 24,297,808 |
| 15 min | 67.6560 | 97.4195 | 73.9590 | 24,297,808 |
| 30 min | 69.2071 | 98.1435 | 74.8824 | 24,297,808 |

Artifacts:

- Checkpoints: `checkpoints/ddp_2gpu/best_rf_gnn_dynamic_sampler.pth`, `checkpoints/ddp_2gpu/rf_gnn_dynamic_sampler_last.pth`
- Metrics: `metrics/baseline_metrics.csv`, `metrics/model_metrics.csv`, `metrics/scaling_metrics.csv`, `metrics/ablation_metrics.csv`
- Plots: `plots/topology.png`, `plots/prediction_vs_truth_station_715898.png`, `plots/error_heatmap.png`
- Prediction CSV: `predictions/prediction_samples_station_715898.csv`

Interpretation:

- This is the strongest full-scale main-model training evidence so far: 10 epochs over the full `162M` timestamp-station-pair slice, with full held-out eval and reproducible artifacts.
- Compared with the 1-epoch fullproof, it provides a real longer-training curve and a full-test GraphSAGE result rather than a sampled model eval.
- The model beats historical average on MAE/RMSE and beats last-value on RMSE, but it does not beat last-value on MAE/MAPE or historical average on MAPE.
- Use this as full-scale training/evaluation evidence and as evidence that GraphSAGE reduces large squared errors; do not describe it as a blanket accuracy win.
