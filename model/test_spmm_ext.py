import os
import unittest

import torch


os.environ.setdefault("SPMM_EXT_DISABLE_JIT", "1")


class TestSpmmFallback(unittest.TestCase):
    def test_cpu_fallback_scatter_add(self):
        from model.spmm_ext import spmm

        index = torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long)
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        out = spmm(index, src, out_size=3)

        expected = torch.tensor([[4.0, 6.0], [5.0, 6.0], [0.0, 0.0]])
        torch.testing.assert_close(out, expected)


class TestDynConvAggregation(unittest.TestCase):
    def test_mean_uses_valid_neighbor_count_not_padded_width(self):
        from model.dyn_conv_final import GraphSAGEDynConvFinal

        conv = GraphSAGEDynConvFinal(in_dim=2, out_dim=2, aggr="mean")
        with torch.no_grad():
            conv.lin.weight.copy_(torch.eye(2))
            conv.lin.bias.zero_()

        x = torch.tensor([[10.0, 10.0], [2.0, 4.0], [6.0, 8.0]])
        nbr_idx = torch.tensor([[1, -1], [0, 2], [-1, -1]], dtype=torch.long)

        out = conv(x, nbr_idx)

        expected = torch.tensor([[2.0, 4.0], [8.0, 9.0], [0.0, 0.0]])
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
