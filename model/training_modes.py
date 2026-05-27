#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared training/evaluation modes for Evidence Pack experiments."""

import torch

FEATURE_ABLATION_NONE = "none"
FEATURE_ABLATION_WITHOUT_WEATHER = "without_weather"
FEATURE_ABLATION_CHOICES = (FEATURE_ABLATION_NONE, FEATURE_ABLATION_WITHOUT_WEATHER)

GRAPH_MODE_FULL = "full"
GRAPH_MODE_SELF_LOOP = "self_loop"
GRAPH_MODE_CHOICES = (GRAPH_MODE_FULL, GRAPH_MODE_SELF_LOOP)

# X_ext feature layout from data_process/step70_make_xy_weather.py:
# flow, occupancy, speed, tavg, pcpn, weekend.
WEATHER_FEATURE_INDICES = (3, 4)


def _chunked(values, chunk_size):
    values = list(values)
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def compute_weather_train_mean(X, train_times, chunk_size=512):
    """Compute tavg/pcpn means from train timestamps only."""
    train_times = list(map(int, train_times))
    if not train_times:
        raise ValueError("Cannot compute weather ablation mean without train times")

    device = X.device if hasattr(X, "device") else torch.device("cpu")
    sums = torch.zeros(len(WEATHER_FEATURE_INDICES), dtype=torch.float64, device=device)
    count = 0
    for chunk in _chunked(train_times, max(1, int(chunk_size))):
        values = X[torch.as_tensor(chunk, dtype=torch.long, device=device)]
        values = values[:, :, list(WEATHER_FEATURE_INDICES)].to(dtype=torch.float64)
        sums += values.sum(dim=(0, 1))
        count += int(values.shape[0] * values.shape[1])
    if count <= 0:
        raise ValueError("Weather ablation selected zero train samples")
    return (sums / count).to(dtype=X.dtype)


def apply_feature_ablation_(X, train_times, feature_ablation, values=None):
    """Apply in-place feature ablation and return reproducibility metadata."""
    if feature_ablation == FEATURE_ABLATION_NONE:
        return {"feature_ablation": FEATURE_ABLATION_NONE}
    if feature_ablation != FEATURE_ABLATION_WITHOUT_WEATHER:
        raise ValueError(f"Unsupported feature_ablation={feature_ablation!r}")

    if values and "weather_mean" in values:
        mean = torch.as_tensor(values["weather_mean"], dtype=X.dtype, device=X.device)
    else:
        mean = compute_weather_train_mean(X, train_times)
    if mean.numel() != len(WEATHER_FEATURE_INDICES):
        raise ValueError(f"Expected {len(WEATHER_FEATURE_INDICES)} weather means, got {mean.numel()}")

    X[:, :, list(WEATHER_FEATURE_INDICES)] = mean.view(1, 1, -1)
    return {
        "feature_ablation": FEATURE_ABLATION_WITHOUT_WEATHER,
        "weather_feature_indices": list(WEATHER_FEATURE_INDICES),
        "weather_mean": [float(v) for v in mean.detach().cpu().tolist()],
        "weather_mean_source": "train_split",
    }


def self_loop_edge_index(num_nodes):
    nodes = torch.arange(int(num_nodes), dtype=torch.long)
    return torch.stack([nodes, nodes], dim=0).contiguous()


def apply_graph_mode(edge_index, num_nodes, graph_mode):
    if graph_mode == GRAPH_MODE_FULL:
        return edge_index.long().contiguous()
    if graph_mode == GRAPH_MODE_SELF_LOOP:
        return self_loop_edge_index(num_nodes)
    raise ValueError(f"Unsupported graph_mode={graph_mode!r}")


def ablation_label(feature_ablation=FEATURE_ABLATION_NONE, graph_mode=GRAPH_MODE_FULL):
    if feature_ablation == FEATURE_ABLATION_WITHOUT_WEATHER:
        return FEATURE_ABLATION_WITHOUT_WEATHER
    if graph_mode == GRAPH_MODE_SELF_LOOP:
        return "without_graph_neighbors"
    return "full"
