# model/dyn_conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGEDynConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim,  out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim,  out_dim, bias=True)

    def forward(self, x, nbr_idx):
        # x: [N, F_in]; nbr_idx: [N, K] (-1 = padding)
        N, K = nbr_idx.size()
        mask = nbr_idx < 0                       # [N, K]
        idx  = nbr_idx.clone()
        idx[mask] = 0                            # 防止越界
        h_nei = x[idx]                           # [N, K, F_in]
        h_nei[mask] = 0                          # padding 置零
        cnt = (~mask).sum(dim=1, keepdim=True).clamp(min=1).float()  # [N,1]
        neigh_mean = h_nei.sum(dim=1) / cnt      # [N, F_in]

        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return F.relu(out)
