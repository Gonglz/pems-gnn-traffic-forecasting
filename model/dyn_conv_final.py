# model/dyn_conv_final.py

import torch
from torch import nn
from.spmm_ext import spmm

class GraphSAGEDynConvFinal(nn.Module):
    """
    Dynamic RF GraphSAGE Conv using custom SpMM kernel.
    input:
      x       [N, F]: note
      nbr_idx [N, K]: note (-1 note)
    output:
      out     [N, out_dim]: noteresult
    """
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        assert aggr in ('mean', 'sum')
        self.aggr = aggr

    def forward(self, x, nbr_idx):
        """
        x: Tensor [N, F]
        nbr_idx: LongTensor [N, K], note, -1 note
        """
        N, F_size = x.size()
        K = nbr_idx.size(1)
        device = x.device

        # 1) note
        valid = (nbr_idx >= 0)
        nbr_idx = nbr_idx.clone()
        nbr_idx[~valid] = 0  # note 0, notecompute(note 0)

        # 2) note row, col note
        row = torch.arange(N, device=device).view(-1, 1).expand(-1, K).reshape(-1)
        col = nbr_idx.reshape(-1)

        # 3) note src note
        src = x[col]  # [N*K, F_size]

        # 4) note: mean note sum
        mask_flat = valid.view(-1).to(x.dtype)
        if self.aggr == 'mean':
            valid_counts = valid.sum(dim=1).clamp_min(1).to(x.dtype)
            weights = mask_flat / valid_counts.repeat_interleave(K)
            src = src * weights.unsqueeze(-1)
        else:  # sum
            src = src * mask_flat.unsqueeze(-1)

        # 5) note AMP half note: note float16, note float32
        orig_dtype = src.dtype
        if orig_dtype == torch.float16:
            src = src.float()

        # 6) note SpMM kernel (sum note)
        out = spmm(
            index   = torch.stack([row, col], dim=0),
            src     = src,
            out_size= N
        )  # [N, F_size]

        # 7) note half, note
        if orig_dtype == torch.float16:
            out = out.half()

        # 8) note
        return self.lin(out)
