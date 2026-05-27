import os
import warnings

import torch
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(__file__)
_spmm_ext = None
_spmm_load_error = None


def _load_spmm_ext():
    """Load the optional CUDA extension on first CUDA use."""
    global _spmm_ext, _spmm_load_error
    if _spmm_ext is not None:
        return _spmm_ext
    if _spmm_load_error is not None:
        return None
    if os.environ.get("SPMM_EXT_DISABLE_JIT", "").lower() in {"1", "true", "yes", "y"}:
        _spmm_load_error = RuntimeError("SPMM extension JIT disabled by SPMM_EXT_DISABLE_JIT")
        return None

    try:
        _spmm_ext = load(
            name="spmm_ext",
            sources=[os.path.join(_this_dir, "spmm_kernel.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=os.environ.get("SPMM_EXT_VERBOSE", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain.
        _spmm_load_error = exc
        warnings.warn(
            f"Falling back to torch.index_add_ because custom SpMM failed to load: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return _spmm_ext


def _torch_spmm(row, src, out_size):
    out = src.new_zeros((out_size, src.size(1)))
    if row.numel() > 0:
        out.index_add_(0, row, src)
    return out

def spmm(index, src, out_size):
    """
    index: LongTensor of shape [2, E] (row, col)
    src:   Tensor of shape [E, F]
    out_size: int number of target nodes (rows)
    returns: Tensor [out_size, F]
    """
    if index.dim() != 2 or index.size(0) != 2:
        raise ValueError(f"index must have shape [2, E], got {tuple(index.shape)}")
    if src.dim() != 2:
        raise ValueError(f"src must have shape [E, F], got {tuple(src.shape)}")

    row, col = index.long()
    row = row.to(device=src.device, non_blocking=True).contiguous()
    col = col.to(device=src.device, non_blocking=True).contiguous()
    src = src.contiguous()

    if src.device.type == "cuda" and src.dtype in {torch.float32, torch.float64}:
        ext = _load_spmm_ext()
        if ext is not None:
            return ext.spmm_cuda(row, col, src, out_size)
    return _torch_spmm(row, src, out_size)
