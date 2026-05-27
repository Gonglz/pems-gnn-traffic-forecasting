# Evidence Plan for PeMS Traffic Forecasting System

生成时间：2026-05-25

目标：补齐简历和面试需要的量化证据链，让项目从“代码整理完成”变成“有可复现实验结果的真实 ML 系统”。

## P0: 必做证据

### P0-1 中样本 benchmark

先不要直接重跑正式 162M 行全量数据。建议先固定一个中样本：

- 时间范围：1 天、1 周或 1 个月，优先选 1 周。
- station 范围：固定一组可复现 station list。
- split：固定 train/val/test 时间切分。
- 输出目录：`/scratch2/lgong1/finalproject_gpu_parallel_test/<timestamp>/`

需要记录：

| Model / Setup | MAE | RMSE | MAPE | Train Time | GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Last-value baseline |  |  |  | - | - |
| Historical average |  |  |  | - | - |
| MLP / RF baseline |  |  |  |  |  |
| GraphSAGE + weather |  |  |  |  |  |
| GraphSAGE + weather + DDP |  |  |  |  |  |

### P0-2 Ablation

做 3 个小消融，目标是解释系统设计为什么有用：

| Experiment | Purpose | Metrics |
| --- | --- | --- |
| without weather | 验证 station-aligned weather 是否提升预测 | MAE/RMSE/MAPE |
| without graph neighbors | 验证拓扑邻居是否提升预测 | MAE/RMSE/MAPE |
| simple ffill vs robust fill | 验证异常/缺失补全策略是否重要 | MAE/RMSE/MAPE + bad rows count |

面试表达目标：

> Weather alignment and topology-aware neighbor sampling improved prediction accuracy compared with naive temporal baselines.

具体提升幅度等实验结果出来后再填。

### P0-3 1 GPU vs 2 GPU DDP scaling

当前代码已经修复 DDP 下重复 time/node 采样问题。下一步需要量化训练扩展性：

| Setup | Epoch Time | Speedup | GPU Utilization | GPU Memory | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 GPU |  | 1.0x |  |  | baseline |
| 2 GPU DDP |  |  |  |  | no duplicate time/node shards |

注意：不要假设线性加速。真实结果是多少就记录多少。

### P0-4 三张可视化

建议保留 3 张图：

1. 传感器拓扑图：station 节点 + edge。
2. 预测 vs 真实曲线：某个 station 某一天 flow 或 speed。
3. 误差热力图：station × time，或 hour-of-day × error。

这些图能让项目从“脚本很多”变成“系统效果可解释”。

## P1: 可选增强

### P1-1 正式全量重跑

如果时间允许，再跑正式全量。目标不是把模型调到最好，而是拿到工程规模证据：

- 全量数据处理耗时
- 训练数组生成耗时
- 单卡训练耗时
- 双卡 DDP 训练耗时
- 峰值显存
- test MAE/RMSE/MAPE

### P1-2 小模型改进

最多选 1-2 个，不要同时堆太多：

- time-of-day / day-of-week embedding
- missing-value mask feature
- distance/road-length edge weight
- stronger baseline：STGCN、DCRNN 或 Graph WaveNet 中的一个

优先级低于 P0，因为当前项目更缺证据链，不缺复杂模型。

### P1-3 历史脚本归档

把旧路径移动到 `archive/` 或在文件头加 deprecated 注释：

- `model/train_final.py`
- `model/train_faster.py`
- `data_process/step40_*`
- 其他未纳入 canonical workflow 的实验脚本

这对简历不是直接加分，但会提升 GitHub 可读性。

## P2: 暂不优先 custom SpMM

除非目标岗位转向 GPU infra / CUDA / ML systems，否则不建议把 custom SpMM 作为下一阶段重点。

原因：

- 会和另一个 systems/HPC 性能优化项目重复。
- 当前项目更强的是真实数据 pipeline、图节点对齐、时空建模和 DDP correctness。
- custom SpMM 投入高，如果没有显著 speedup，很难形成高质量简历证据。

更成熟的面试说法：

> I considered a custom sparse-matrix path, but after stabilizing the end-to-end workflow I prioritized PyG's optimized NeighborSampler/SAGEConv path, because the larger risks were data correctness, graph-node alignment, and distributed sampling correctness.

## 推荐输出结构

```text
/scratch2/lgong1/finalproject_gpu_parallel_test/<timestamp>/
  config/
    data_slice.json
    train_config.json
  metrics/
    baseline_metrics.csv
    model_metrics.csv
    scaling_metrics.csv
  plots/
    topology.png
    prediction_vs_truth_station_<id>.png
    error_heatmap.png
  checkpoints/
    single_gpu/
    ddp_2gpu/
```

## 当前工具入口

Evidence Pack 入口已经集成到：

```bash
cd /scratch/lgong1/finalproject

/scratch/lgong1/envs/traffic-env/bin/python -m model.evidence_pack run \
  --source-data-dir /scratch/lgong1/finalproject/pems_data
```

默认会创建 `/scratch2/lgong1/finalproject_gpu_parallel_test/<timestamp>/`，切出
`2025-01-01T00:00:00 <= timestamp < 2025-01-08T00:00:00` 的 1 周样本，并使用
全部 `step62_neighbors.pkl:graph_nodes` 保持图节点对齐。

完整单卡 + 双卡训练使用默认参数：

```bash
/scratch/lgong1/envs/traffic-env/bin/python -m model.evidence_pack run \
  --source-data-dir /scratch/lgong1/finalproject/pems_data \
  --epochs 3 \
  --batch-size 1024 \
  --num-workers 4 \
  --max-train-times 512 \
  --max-val-times 128
```

只验证切片和 baseline，不训练：

```bash
/scratch/lgong1/envs/traffic-env/bin/python -m model.evidence_pack run \
  --source-data-dir /scratch/lgong1/finalproject/pems_data \
  --output-root /scratch2/lgong1/finalproject_gpu_parallel_test/prepare-check-20260525-0615 \
  --skip-training \
  --skip-rf
```

## 最终可写进简历的目标结果

等 P0 数据补齐后，简历 bullet 可以写成：

- Built an end-to-end spatiotemporal GNN forecasting pipeline on PeMS traffic sensor data, including robust anomaly imputation, sensor graph construction, station-aligned weather features, and PyG GraphSAGE training.
- Prevented data leakage from masked anomalous values and fixed graph/data station-order mismatches across `X/Y/sids/timestamps` and graph nodes.
- Implemented DDP training with non-overlapping time/node shards across GPUs; measured 1 GPU vs 2 GPU training speedup on a medium-scale PeMS benchmark.
- Improved MAE/RMSE/MAPE over last-value and historical-average baselines by `<fill after benchmark>%`.
