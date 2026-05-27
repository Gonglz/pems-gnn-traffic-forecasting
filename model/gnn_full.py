# model/gnn_full.py
import torch.nn as nn
from.dyn_conv import GraphSAGEDynConv
import torch.nn.functional as F


class MultiHeadRFGraphSAGEDyn(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        # note Dynamic SAGEConv
        self.convs5  = nn.ModuleList([
            GraphSAGEDynConv(in_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.convs15 = nn.ModuleList([
            GraphSAGEDynConv(in_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.convs30 = nn.ModuleList([
            GraphSAGEDynConv(in_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = dropout
        # note
        self.head5  = nn.Linear(hidden_dim, 1)
        self.head15 = nn.Linear(hidden_dim, 1)
        self.head30 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x  # [N, F]
        # 5min branch
        h5 = x
        for conv in self.convs5:
            h5 = conv(h5, data.nbr5.to(h5.device))
            h5 = F.dropout(h5, p=self.dropout, training=self.training)
        # 15min branch
        h15 = x
        for conv in self.convs15:
            h15 = conv(h15, data.nbr15.to(h15.device))
            h15 = F.dropout(h15, p=self.dropout, training=self.training)
        # 30min branch
        h30 = x
        for conv in self.convs30:
            h30 = conv(h30, data.nbr30.to(h30.device))
            h30 = F.dropout(h30, p=self.dropout, training=self.training)

        return self.head5(h5), self.head15(h15), self.head30(h30)
