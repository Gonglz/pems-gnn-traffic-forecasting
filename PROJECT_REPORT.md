# PeMS Spatiotemporal Traffic Forecasting System Report

生成时间：2026-05-25

项目根目录：

- 主项目：`/scratch/lgong1/finalproject`
- smoke/中间验证输出：`/scratch2/lgong1/finalproject_smoke/20260525-041635`
- 本轮修改前代码备份：`/scratch2/lgong1/finalproject_backups/20260525-041635-code.tar.gz`
- 推荐 Python 环境：`/scratch/lgong1/envs/traffic-env/bin/python`

## 1. 项目定位

这个项目更准确的定位是：**面向真实 PeMS 交通传感器数据的时空图神经网络预测系统**。

它覆盖了从真实交通数据清洗到图神经网络训练和推理的完整 ML engineering 链路：

1. 异常/缺失交通观测补全，避免 masked raw anomalous value 泄漏。
2. 基于 PeMS station metadata 构建道路传感器拓扑。
3. 按 station/grid 对齐天气特征，避免 timestamp-only weather 广播。
4. 对齐 `X/Y/sids/timestamps` 与图节点，保证训练数据和拓扑一致。
5. 使用 PyTorch Geometric 的 GraphSAGE + NeighborSampler 做时空图预测。
6. 修复多 GPU DDP 训练中的 time/node 重复采样和多 head forward 同步问题。
7. 支持真实 PeMS station ID 和内部 graph node index 两种推理入口。

这个项目不应作为“底层 kernel 加速项目”来讲。更适合的主标签是：

- ML Engineering
- 图神经网络 / Spatiotemporal Forecasting
- 真实数据 pipeline
- PyTorch / PyG / DDP
- 可复现训练与推理系统

GPU 和 DDP 是训练基础设施能力点，但不是本项目主叙事。这样可以和另一个 Python -> C / CPU 多核 / CUDA 性能优化项目形成互补：一个讲底层性能优化，一个讲真实数据上的大规模 ML 系统。

## 2. 当前主流程

当前推荐维护一条 canonical workflow：

1. 数据补全：`data_process/step50_Fill.py` -> `data_process/step51_refill.py`
2. 补全检查：`data_process/step50_fillCheck.py`
3. 拓扑构建：`data_process/step52_buildTopo.py`
4. 邻域预计算：`data_process/step62_precompute_neighbors.py`
5. 训练数组生成：`pems_data/step70_make_xy_weather.py`
6. 模型训练：`python -m model.train_sampler`
7. 模型推理：`python -m model.predict_and_plot`

可执行命令整理在：

- `/scratch/lgong1/finalproject/CANONICAL_WORKFLOW.md`

第一轮整理没有覆盖正式大产物，也没有覆盖正式 checkpoint。所有测试 checkpoint 和输出都在 `/scratch2/lgong1/finalproject_smoke/20260525-041635`。

## 3. 当前入口与历史路径

### 3.1 当前数据处理入口

- `data_process/step50_Fill.py`
- `data_process/step51_refill.py`
- `data_process/step50_fillCheck.py`
- `data_process/step52_buildTopo.py`
- `data_process/step62_precompute_neighbors.py`
- `pems_data/step70_make_xy_weather.py`

### 3.2 当前模型入口

- `model/dataset_full.py`
- `model/gnn_final.py`
- `model/train_sampler.py`
- `model/predict_and_plot.py`

### 3.3 历史/实验路径

这些文件暂时不作为当前生产入口：

- `model/train_final.py`
- `model/train_faster.py`
- `data_process/step40_*`
- `model/spmm_ext.py`

原因：这些路径和当前验证通过的 PyG `SAGEConv + NeighborSampler` 主线不一致，且没有在本轮 smoke 中纳入端到端验证。`model/spmm_ext.py` 属于 custom SpMM 实验路径，保留为可参考实现，但不建议作为当前项目叙事重点。

## 4. 本轮关键修复

### 4.1 数据补全链

修改文件：

- `data_process/step50_Fill.py`
- `data_process/step51_refill.py`
- `data_process/step50_fillCheck.py`
- `data_process/test/test_worker_a_fill_chain.py`

修复内容：

- `step50_Fill.py` 不再把 station metadata 中的 KNN 下标当成 partition 行号使用。
- local fill 改为按同一 `timestamp` 下的 `neighbor_station_id` 对齐邻居值。
- masked raw 特征会先置空，避免异常原值进入补全过程。
- temporal fill 保持 timestamp 为 int64，避免 float32 epoch 精度问题。
- 修复 cudf 真实运行时 `groupby.apply(lambda group: group.ffill().bfill())` 的 numba 编译失败，改成显式 pandas/cudf 兼容实现。
- `step51_refill.py` 禁止用 masked raw anomalous value 回填。
- `step51_refill.py` 合并前先聚合去重，输出行数必须等于上游唯一 `(timestamp, station_id)` 行数。
- `step50_fillCheck.py` 删除 import-time debug 读取，支持 step50/step51 两种 schema。

### 4.2 特征、天气、拓扑链

修改文件：

- `pems_data/step70_make_xy_weather.py`
- `data_process/step52_buildTopo.py`
- `data_process/step62_precompute_neighbors.py`
- `data_process/test/test_worker_b_generation.py`

修复内容：

- `step70_make_xy_weather.py` 的 weather 改成按 station/grid 对齐。
- 不再 `drop_duplicates("timestamp")` 后把同一个天气广播到所有 station。
- 生成 `timestamps.npy`。
- `Y.npy` 明确保存为 `float32`。
- `X/Y/sids` 可按 `step52_graph_nodes.pkl` 过滤和排序，保证与图节点一致。
- `step52_buildTopo.py` 输出 `step52_graph_nodes.pkl`，明确记录缺坐标被剔除的 station。
- `step62_precompute_neighbors.py` 所有写文件逻辑移入 `main()`，避免 import-time side effect。
- `step62_precompute_neighbors.py` 对 NaN/非法 `length` 使用坐标距离或默认长度 fallback。

### 4.3 训练/推理链

修改文件：

- `model/dataset_full.py`
- `model/gnn_final.py`
- `model/train_sampler.py`
- `model/predict_and_plot.py`
- `model/test_worker_c.py`

修复内容：

- `dataset_full.py` 支持 `data_dir` 和路径覆盖，支持 `timestamps.npy`。
- `dataset_full.py` 支持真实 station ID 与内部 node index 互相映射。
- `dataset_full.py` 兼容旧版 `step62_neighbors.pkl` 和新版带 `graph_nodes` 的 payload。
- `train_sampler.py` 不再只训练 `t0=0`，改为采样多个 time index。
- `train_sampler.py` 验证集使用 held-out 时间段。
- DDP 下按 rank 切分 time index 和 seed nodes，避免双卡完全重复。
- 修复 DDP 中连续三次 head-specific forward 导致的 `Expected to mark a variable ready only once`。
- 当前 DDP forward 会一次性跑三个 head，然后合并 loss backward。
- eval 阶段使用 unwrapped module 做本地 forward，避免某些 rank 没有 held-out 时间片时 DDP eval hang。
- `predict_and_plot.py` 支持真实 PeMS `--station-id` 和内部 `--node-idx`。
- `predict_and_plot.py` 支持 `--time` 通过 `timestamps.npy` 定位。
- 修复实时 override 中 occupancy/speed 的特征列顺序。

## 5. 已验证结果

### 5.1 静态和单元测试

远端通过：

```bash
/scratch/lgong1/envs/traffic-env/bin/python -m py_compile \
  data_process/step50_Fill.py \
  data_process/step51_refill.py \
  data_process/step50_fillCheck.py \
  pems_data/step70_make_xy_weather.py \
  data_process/step52_buildTopo.py \
  data_process/step62_precompute_neighbors.py \
  model/train_sampler.py \
  model/predict_and_plot.py \
  model/dataset_full.py \
  model/gnn_final.py \
  model/test_worker_c.py
```

远端单元测试：

- `data_process/test`：18 tests OK
- `model.test_worker_c`：6 tests OK

测试过程中有 pandas/numba/cudf 依赖版本 warning，但没有失败。

### 5.2 小样本端到端 smoke

smoke 路径：

- `/scratch2/lgong1/finalproject_smoke/20260525-041635`

构造样本：

- raw rows：40
- masked rows：2
- station：4 个，其中 1 个缺坐标用于验证拓扑剔除
- weather：按 station 构造不同值，用于验证不再广播

验证结果：

- step50 multi-GPU sample-mode：通过
- step51 输出行数：40，未膨胀
- masked rows still missing：0
- masked raw 异常值未被用于回填
- step52 剔除缺坐标 station 104，保留 `[101, 102, 103]`
- step62 输出 graph nodes 和 neighbors：3 nodes
- step70 输出：
  - `X_ext.npy` shape `(10, 3, 6)`，dtype `float32`
  - `Y.npy` shape `(10, 3)`，dtype `float32`
  - `sids.npy` 为 `[101, 102, 103]`
  - `timestamps.npy` 长度 10
- 同一 timestamp 下 weather 三站不同，确认没有 timestamp-only 广播。

### 5.3 训练和推理 smoke

单卡训练 smoke：

- 结果：forward/backward 成功，loss finite，checkpoint 已写入 smoke 目录。

双卡 DDP smoke：

- 命令使用 `torchrun --standalone --nproc_per_node=2`
- 结果：forward/backward 成功，loss finite，checkpoint 已写入 smoke 目录。
- rank 分片检查：
  - rank0 train times `[0, 2]`，nodes `[0, 2]`
  - rank1 train times `[1]`，nodes `[1]`
- 说明：双卡不是重复跑同一批数据。

推理 smoke：

- 使用真实 station id：`--station-id 101 --time-idx 0`，通过。
- 使用内部 node idx：`--node-idx 0 --time "2024-01-01 00:00:00"`，通过。
- 输出图：
  - `/scratch2/lgong1/finalproject_smoke/20260525-041635/plots/pred_station_101_timeidx0.png`
  - `/scratch2/lgong1/finalproject_smoke/20260525-041635/plots/pred_node0_time.png`

## 6. Medium-scale Evidence Pack (P0)

正式 P0 evidence 目录：

- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936`

详细结果记录在：

- `/scratch/lgong1/finalproject/EVIDENCE_RESULTS.md`

数据切片：

- 时间范围：`2025-01-01 00:00:00` 到 `2025-01-07 23:55:00`
- 时间步：`2016` 个 5 分钟 timestamp
- 站点：`4883` 个 graph-aligned PeMS station
- 特征：`flow_interp`, `occupancy_interp`, `speed_interp`, `tavg`, `pcpn`, `is_weekend`
- 目标：`flow_interp`
- 预测 horizon：`t+5`, `t+15`, `t+30` minutes
- 评估 split：held-out final 15% of valid time indices
- 数组：`X_ext.npy=(2016, 4883, 6)`, `Y.npy=(2016, 4883)`

核心指标（all horizons）：

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 23.2402 | 50.8365 | 39.7728 | naive temporal baseline |
| Historical average | 21.6623 | 39.6356 | 37.7809 | train-split time-of-day baseline |
| GraphSAGE + weather, 1 GPU | 27.9084 | 48.1118 | 50.6426 | 3 epoch checkpoint |
| GraphSAGE + weather, 2 GPU DDP | 26.9821 | 48.0007 | 49.0806 | 3 epoch checkpoint |

Scaling：

| Setup | Epoch Time Sec | Speedup | Peak GPU Memory MB | Total Train Time Sec |
|---|---:|---:|---:|---:|
| 1 GPU | 274.4828 | 1.0000x | 23.8276 | 1019.3152 |
| 2 GPU DDP | 176.4142 | 1.5559x | 24.0918 | 658.1723 |

可视化产物：

- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/topology.png`
- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/prediction_vs_truth_station_715898.png`
- `/scratch2/lgong1/finalproject_gpu_parallel_test/p0-20260525-062936/plots/error_heatmap.png`

结果解释：这个 P0 run 已经可以作为可复现实验包和 DDP scaling 证据，但 3 epoch GraphSAGE checkpoint 还没有在 MAE/MAPE 上超过 historical-average baseline。当前不能声称 accuracy improvement，应表述为完成 closed-loop benchmark、记录 baseline/model/scaling/visualization，并把调参、RF/MLP、真实 ablation 放到下一轮。

`metrics/ablation_metrics.csv` 中 `without_weather`、`without_graph_neighbors`、`simple_ffill` 仍是 `notes=not_available` 占位，不应作为 ablation 结论。

## 7. P1 扩展结果

P1 结果详见：

- `/scratch/lgong1/finalproject/EVIDENCE_RESULTS.md`

本轮新增能力：

- `train_sampler.py` 默认使用 `--ddp-shard-mode flat-batch`，按 `(time_idx, seed_node_batch)` 展平分片，保证 1 GPU / 2 GPU 覆盖同等 workload。
- Evidence Pack 支持 `--skip-single`，full-scale proof 可只跑 2 GPU DDP，并在 scaling CSV 中记录 `skipped_by_config`。
- `without_weather` 和 `without_graph_neighbors` 已变成真实训练/eval ablation，不再只是空 CSV 占位。
- eval 会输出 station/timestamp 级预测样例 CSV：`predictions/prediction_samples_station_<id>.csv`。
- 非 TTY 日志下禁用 tqdm 进度条，避免 full-scale DDP stdout pipe 被无换行进度条卡住。

P1 medium tuned run：

- 目录：`/scratch2/lgong1/finalproject_gpu_parallel_test/p1-medium-tune-20260525-112303`
- 配置：1-week slice，10 epochs，2 GPU DDP，`dropout=0.1`，full held-out test split
- GraphSAGE + weather DDP all-horizon：MAE `26.3360`，RMSE `47.6633`，MAPE `49.3620`
- 结论：比 P0 DDP MAE `26.9821` 有改善，但仍没有超过 historical-average baseline `21.6623`，不能写 accuracy improvement。

P1 fair scaling：

- 目录：`/scratch2/lgong1/finalproject_gpu_parallel_test/p1-fair-scaling-20260525-112038`
- 1 GPU epoch：`18.7796s`
- 2 GPU DDP epoch：`13.0925s`
- fair flat-batch speedup：`1.4344x`
- rank0/rank1 各处理 `1,250,048` 个 train time-node pairs。

P1 真实 ablation：

| Ablation | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| `without_weather` | 26.8646 | 47.9663 | 48.8778 | train-split weather mean replacement |
| `without_graph_neighbors` | 27.3594 | 48.1248 | 49.0564 | self-loop-only graph |

P1 full-scale proof：

- 目录：`/scratch2/lgong1/finalproject_gpu_parallel_test/fullproof-20260525-113032`
- 全量 slice：`33177` timestamps，`4883` stations，`162,003,291` timestamp-station pairs
- DDP train workload：`129,575,288` train time-node pairs，rank0/rank1 各 `64,787,644`
- DDP train steps：`132,680` global，`66,340` per rank
- 2 GPU DDP epoch time：`572.7665s`
- total train time：`631.9582s`
- artifacts：checkpoint、baseline/model/scaling/ablation CSV、topology/prediction/error plots、prediction CSV 全部生成
- 结论：这是 scale proof，不是 accuracy claim。

## 8. P2 模型质量结果

P2 只做模型质量，不扩新 infrastructure。它在同一 Evidence Pack 链路上补齐了：

- target/window audit：确认 `X[t]` 是当前时刻输入，目标为 `Y[t+1]`、`Y[t+3]`、`Y[t+6]`。
- train-only station-wise normalization：输入 flow 和三个 horizon target 都只从 train split fit stats，eval 前 inverse transform 回 raw flow。
- temporal encoding：追加 `sin/cos time-of-day`、`sin/cos day-of-week`、`is_holiday`，保留 `is_weekend`。
- station embedding：GraphSAGE 和 MLP/no-graph 都使用 `16` 维 station embedding。
- horizon-specific objective：三个 horizon 分别 normalize target，loss 为 normalized MSE 等权平均。
- MLP/no-graph baseline：同样的特征、split、normalization、metric function，但不使用 graph neighbors。

正式 P2 runs：

- GraphSAGE：`/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-graphsage-20260525-125437`
- MLP/no-graph：`/scratch2/lgong1/finalproject_gpu_parallel_test/p2-4w-mlp-20260525-141756`

P2 数据切片：

- `2025-01-01T00:00:00` 到 `2025-01-28T23:55:00`
- `8064` timestamps，`4883` graph-aligned stations
- `39,376,512` timestamp-station pairs
- full held-out final 15% eval：`17,710,641` all-horizon samples

All-horizon raw-flow metrics：

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 30.7889 | 62.6905 | 46.5187 | same 4-week held-out split |
| Historical average | 41.5614 | 61.4426 | 64.2816 | train-split time-of-day baseline |
| MLP/no-graph + temporal + station embedding | 31.7255 | 49.4285 | 50.2572 | 30 epochs |
| GraphSAGE + temporal + station embedding | 30.7541 | 48.6064 | 47.2580 | 30 epochs |

P2 结论：

- GraphSAGE 在 4-week P2 设定下超过 historical-average baseline 的 MAE/RMSE/MAPE。
- GraphSAGE 超过 MLP/no-graph baseline 的 MAE/RMSE/MAPE，说明 graph neighbor 信息在当前设定下有实测贡献。
- GraphSAGE 只是在 MAE/RMSE 上超过 last-value，MAPE 仍略差于 last-value，所以不要写 blanket “beats all baselines on all metrics”。
- 训练脚本记录的 peak GPU memory 仍只作为 internal/script-recorded 值，不放进简历主 bullet。

P2 full-scale GraphSAGE 10 epoch run：

- 目录：`/scratch2/lgong1/finalproject_gpu_parallel_test/full-p2-graphsage-10ep-20260525-152216`
- 全量 slice：`33177` timestamps，`4883` stations，`162,003,291` timestamp-station pairs
- 训练：10 epochs，2 GPU DDP，`113,378,377` train time-node pairs/epoch
- best val normalized MSE：`1.4332`
- total train time：`6767.0505s`
- full held-out eval：`72,893,424` all-horizon samples，无 eval cap

Full-scale all-horizon raw-flow metrics：

| Model | MAE | RMSE | MAPE | Notes |
|---|---:|---:|---:|---|
| Last value | 62.8537 | 116.6103 | 55.9902 | full held-out split |
| Historical average | 77.4977 | 106.8358 | 53.9472 | train-split time-of-day baseline |
| GraphSAGE P2 full-scale, 10 epochs | 67.9174 | 97.6607 | 74.2615 | full held-out split |

Full-scale 结论：这是目前最强的全量主模型训练/评估证据。GraphSAGE 在全量 held-out split 上降低 RMSE，并在 MAE/RMSE 上超过 historical average，但没有在 MAE/MAPE 上超过 last-value，也没有在 MAPE 上超过 historical average。因此这条适合写成 full-scale train/eval proof 和 large-error reduction evidence，不适合写成“全面超过所有 baseline”。

详细 P2 结果见：

- `/scratch/lgong1/finalproject/EVIDENCE_RESULTS.md`

## 9. 简历/面试讲法

推荐一句话：

> Built an end-to-end spatiotemporal graph neural network forecasting system on real PeMS traffic sensor data, covering robust anomaly/missing-value imputation, sensor graph construction, station-aligned weather features, PyG GraphSAGE/NeighborSampler training, and multi-GPU DDP training without duplicate temporal/node shards.

当前 evidence-safe 简历 bullets：

> Built an Evidence Pack benchmark over a 1-week, 4,883-station PeMS slice, reporting MAE/RMSE/MAPE, 1-GPU vs 2-GPU DDP scaling, and topology/prediction/error visualizations.

> Measured 1.43x fair DDP training speedup on the medium-scale slice using flat `(time_idx, seed-node-batch)` sharding.

> Ran a full-scale 2-GPU 1-epoch proof over 33,177 timestamps, 4,883 stations, and 162M timestamp-station pairs, producing checkpoints, metrics, plots, logs, and prediction samples.

> Added train-only station-wise normalization, temporal encodings, station embeddings, horizon-specific loss, and an MLP/no-graph baseline; on a 4-week slice, GraphSAGE beat historical-average and MLP/no-graph baselines on MAE/RMSE/MAPE.

> Trained the P2 GraphSAGE model for 10 epochs over the full 162M timestamp-station-pair PeMS slice with 2-GPU DDP, producing full held-out metrics, checkpoints, prediction samples, and visualizations.

暂时不要写：

> Improved over all baselines on all metrics.

原因：P2 GraphSAGE 已经超过 historical-average 和 MLP/no-graph，但 last-value 在 MAPE 上仍略好。可以写精确比较，不能写泛化过头的 blanket claim。

推荐主标签：

- ML Engineering
- Graph Neural Networks
- Spatiotemporal Forecasting
- Data Pipeline for Real Sensor Data
- PyTorch / PyG / Distributed Training

可以强调的工程问题：

- 真实 PeMS 数据存在异常值、缺失值、重复行和 schema 不一致。
- 补全过程需要避免 masked anomalous raw values 泄漏进 label/features。
- weather 必须按 station/grid 对齐，不能按 timestamp 广播到全站。
- `X/Y/sids/timestamps` 必须和 graph nodes 保持同一 station order。
- Naive DDP 会让多个 rank 重复或不公平地处理 time/node，需要按 flattened batch 显式 shard。
- 推理接口要支持真实 PeMS station ID，而不是只暴露内部 node index。

避免这样讲：

- 不要把它称为底层 CUDA kernel 项目。
- 不要把 custom SpMM 作为主成果。
- 不要和另一个 Python -> C / CPU 多核 / CUDA 性能优化项目争同一个叙事空间。

## 10. 当前风险点

1. P0/P1 已有量化证据，但不是准确率胜出证据；P2 开始有模型质量证据。
   - 10 epoch tuned GraphSAGE checkpoint 仍没有在 MAE/MAPE 上超过 historical-average baseline。
   - 4-week P2 GraphSAGE 超过 historical-average 和 MLP/no-graph，但没有在 MAPE 上超过 last-value。
   - 对外要讲精确比较，不要一句话概括成“超过所有 baseline”。

2. ablation 已补两个真实实验，但解释要克制。
   - `without_weather` 和 `without_graph_neighbors` 已经是真训练/eval。
   - `simple_ffill` 仍是 deferred evidence。
   - 当前 ablation 只说明短训设置下的相对变化，不能支撑强因果结论。

3. MLP/no-graph baseline 已纳入 P2；RF 仍是可选 P1/P3。
   - Last-value 和 historical-average baseline 已经可用。
   - MLP/no-graph 已经证明 graph neighbors 在 P2 设定下有贡献。
   - RF 可以继续作为传统 ML 对照，但不要阻塞主线。

4. step50 全量性能需要单独压测。
   - 已修 correctness 和 smoke multi-GPU 运行错误。
   - 还需要对正式数据跑中样本/分片测试，记录 wall time、GPU utilization、memory。

5. 历史脚本很多。
   - 已标记推荐入口和历史入口。
   - 后续最好再做一次文件级归档，把历史实验脚本移动到 `archive/` 或在文件头加 deprecated 注释。

## 11. 建议下一步

下一步不是继续堆 custom kernel，而是在 P2 已有模型质量证据上做小步收敛：

1. 复查 P2 prediction plots/error heatmap，找 last-value MAPE 仍强的 station/time pattern。
2. 只在 4-week slice 上尝试少量 model-quality knobs：dropout、hidden dim、station embedding dim、horizon head capacity。
3. RF 可作为传统 ML 对照，但不能阻塞当前 GraphSAGE/MLP 证据链。
4. 若要对外讲 memory，另跑一次 `max_memory_reserved()` + `nvidia-smi` logging 的确认 pass。
5. 将历史实验脚本做 archive/deprecated 标记，降低项目入口噪音。

详细计划见：

- `/scratch/lgong1/finalproject/EVIDENCE_PLAN.md`

当前 P0 结果见：

- `/scratch/lgong1/finalproject/EVIDENCE_RESULTS.md`
