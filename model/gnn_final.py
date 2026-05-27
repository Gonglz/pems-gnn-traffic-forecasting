#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/gnn_final.py

Multi-Head RF-GraphSAGE with dynamic RF note.
note head note(5/15/30min), note GraphSAGE
noteoutputnote.forward() note sample_forward(), note DDP wrapper
notefirstnote.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


HEADS = ("5min", "15min", "30min")


class MultiHeadRFGraphSAGEDyn(nn.Module):
    """
    Multi-Head RF-GraphSAGE with dynamic RF note.
    note head note(5/15/30min), note GraphSAGE
    noteoutputnote.
    forward() note sample_forward(), note full-graph forward.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_nodes: int = None,
        station_embedding_dim: int = 0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        if station_embedding_dim and num_nodes is None:
            raise ValueError("num_nodes is required when station_embedding_dim > 0")

        self.num_layers = num_layers
        self.dropout    = dropout
        self.station_embedding_dim = int(station_embedding_dim or 0)
        if self.station_embedding_dim:
            self.station_embedding = nn.Embedding(int(num_nodes), self.station_embedding_dim)
        else:
            self.station_embedding = None
        conv_in_dim = in_dim + self.station_embedding_dim

        # note conv layers note
        self.convs = nn.ModuleDict({
            wnd: nn.ModuleList([
                # note:  in_dim -> hidden_dim, note hidden_dim -> hidden_dim
                SAGEConv(conv_in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            for wnd in HEADS
        })
        self.heads = nn.ModuleDict({
            wnd: nn.Linear(hidden_dim, 1)
            for wnd in HEADS
        })

    def _append_station_embedding(self, x, node_ids):
        if self.station_embedding is None:
            return x
        if node_ids is None:
            raise ValueError("node_ids are required for station embedding")
        node_ids = node_ids.to(device=x.device, dtype=torch.long)
        return torch.cat([x, self.station_embedding(node_ids)], dim=1)

    def sample_forward(
        self,
        x: torch.Tensor,
        adjs: list,
        head: str,
        node_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        notefirstnote.
        x:   [num_sub_nodes, in_dim]  inputnote
        adjs: list of (edge_index, e_id, size) tuples from NeighborSampler
              note = num_layers, note [hop_1, hop_2, …]
        head: '5min' | '15min' | '30min'
        note: [batch_size, 1] note
        """
        convs = self.convs[head]
        lin   = self.heads[head]
        x = self._append_station_embedding(x, node_ids)

        # note adjs note: adjs[0] note, adjs[1] note
        for i, (edge_index, _, size) in enumerate(adjs):
            # size = [num_source_nodes, num_target_nodes]
            x_src = x[:size[0]]   # note source note
            x_dst = x[:size[1]]   # batch note target note
            # noterows GraphSAGE note
            x = convs[i]((x_src, x_dst), edge_index)
            # note dropout
            if i!= self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # x notegoalnote, noteoutput
        return lin(x)

    def forward(
        self,
        x: torch.Tensor = None,
        adjs: list = None,
        head: str = None,
        node_ids: torch.Tensor = None,
        head_batches: dict = None,
    ):
        if head_batches is not None:
            outputs = {}
            for head_name, batch in head_batches.items():
                if len(batch) == 2:
                    batch_x, batch_adjs = batch
                    batch_node_ids = None
                else:
                    batch_x, batch_adjs, batch_node_ids = batch
                outputs[head_name] = self.sample_forward(
                    batch_x,
                    batch_adjs,
                    head_name,
                    node_ids=batch_node_ids,
                )
            return outputs
        if head is None:
            raise ValueError("Specify head or head_batches")
        return self.sample_forward(x, adjs, head, node_ids=node_ids)


class MultiHeadRFMLP(nn.Module):
    """No-graph baseline with the same temporal/station feature interface."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_nodes: int = None,
        station_embedding_dim: int = 0,
    ):
        super().__init__()
        if station_embedding_dim and num_nodes is None:
            raise ValueError("num_nodes is required when station_embedding_dim > 0")
        self.station_embedding_dim = int(station_embedding_dim or 0)
        if self.station_embedding_dim:
            self.station_embedding = nn.Embedding(int(num_nodes), self.station_embedding_dim)
        else:
            self.station_embedding = None
        input_dim = int(in_dim) + self.station_embedding_dim
        depth = max(1, int(num_layers))
        self.heads = nn.ModuleDict()
        for head in HEADS:
            layers = []
            last_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, 1))
            self.heads[head] = nn.Sequential(*layers)

    def _append_station_embedding(self, x, node_ids):
        if self.station_embedding is None:
            return x
        if node_ids is None:
            raise ValueError("node_ids are required for station embedding")
        node_ids = node_ids.to(device=x.device, dtype=torch.long)
        return torch.cat([x, self.station_embedding(node_ids)], dim=1)

    def sample_forward(self, x, head, node_ids=None):
        x = self._append_station_embedding(x, node_ids)
        return self.heads[head](x)

    def forward(self, x=None, head=None, node_ids=None, head_batches=None, **_):
        if head_batches is not None:
            outputs = {}
            for head_name, batch in head_batches.items():
                if len(batch) == 2:
                    batch_x, batch_node_ids = batch
                elif len(batch) == 3:
                    batch_x, _, batch_node_ids = batch
                else:
                    raise ValueError("head_batches values must have length 2 or 3")
                outputs[head_name] = self.sample_forward(
                    batch_x,
                    head_name,
                    node_ids=batch_node_ids,
                )
            return outputs
        if head is None:
            raise ValueError("Specify head or head_batches")
        return self.sample_forward(x, head, node_ids=node_ids)
