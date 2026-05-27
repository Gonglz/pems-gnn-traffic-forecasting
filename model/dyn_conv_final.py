# model/dyn_conv_final.py

import torch
from torch import nn
from .spmm_ext import spmm

class GraphSAGEDynConvFinal(nn.Module):
    """
    Dynamic RF GraphSAGE Conv using custom SpMM kernel.
    输入：
      x       [N, F]       : 节点特征
      nbr_idx [N, K]       : 每个节点的邻居列表 (-1 用来填充)
    输出：
      out     [N, out_dim] : 聚合后线性变换结果
    """
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        assert aggr in ('mean', 'sum')
        self.aggr = aggr

    def forward(self, x, nbr_idx):
        """
        x       : Tensor [N, F]
        nbr_idx : LongTensor [N, K]，邻居索引，-1 表示无效位置
        """
        N, F_size = x.size()
        K = nbr_idx.size(1)
        device = x.device

        # 1) 掩码无效邻居
        valid = (nbr_idx >= 0)
        nbr_idx = nbr_idx.clone()
        nbr_idx[~valid] = 0  # 临时指向 0，不会影响计算（后续乘以权重 0）

        # 2) 构造扁平化的 row, col 索引
        row = torch.arange(N, device=device).view(-1, 1).expand(-1, K).reshape(-1)
        col = nbr_idx.reshape(-1)

        # 3) 提取 src 特征
        src = x[col]  # [N*K, F_size]

        # 4) 加权：mean 或 sum
        mask_flat = valid.view(-1).to(x.dtype)
        if self.aggr == 'mean':
            valid_counts = valid.sum(dim=1).clamp_min(1).to(x.dtype)
            weights = mask_flat / valid_counts.repeat_interleave(K)
            src = src * weights.unsqueeze(-1)
        else:  # sum
            src = src * mask_flat.unsqueeze(-1)

        # 5) 兼容 AMP half 精度：若是 float16，临时升级为 float32
        orig_dtype = src.dtype
        if orig_dtype == torch.float16:
            src = src.float()

        # 6) 调用自写 SpMM kernel （sum 聚合）
        out = spmm(
            index   = torch.stack([row, col], dim=0),
            src     = src,
            out_size= N
        )  # [N, F_size]

        # 7) 若原来是 half，降回去
        if orig_dtype == torch.float16:
            out = out.half()

        # 8) 线性映射
        return self.lin(out)
