#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 256

/**
 * Edge-list scatter-add used by dyn_conv_final.py:
 * out[row[e], f] += src[e, f].
 */
template <typename scalar_t>
__global__ void spmm_cuda_kernel(
    const int64_t* __restrict__ row,   // [NNZ]
    const int64_t* __restrict__ col,   // [NNZ]  (unused here)
    const scalar_t* __restrict__ src,  // [NNZ, F] flattened: src[e * F + f]
    scalar_t* __restrict__ out,        // [N, F]  flattened: out[u * F + f]
    int64_t NNZ,
    int64_t N,
    int64_t F)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = NNZ * F;
    if (idx >= total) return;

    int64_t e = idx / F;
    int64_t f = idx % F;
    int64_t u = row[e];
    if (u >= 0 && u < N) {
        atomicAdd(&out[u * F + f], src[e * F + f]);
    }
}

/**
 * C++ interface called from Python.
 * row, col: LongTensor of shape [NNZ]
 * src:      FloatTensor of shape [NNZ, F]
 * N:        number of output rows
 */
torch::Tensor spmm_cuda(
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor src,
    int64_t N)
{
    TORCH_CHECK(row.is_cuda(), "row must be a CUDA tensor");
    TORCH_CHECK(col.is_cuda(), "col must be a CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(row.scalar_type() == torch::kInt64, "row must be int64");
    TORCH_CHECK(col.scalar_type() == torch::kInt64, "col must be int64");
    TORCH_CHECK(src.dim() == 2, "src must have shape [NNZ, F]");
    TORCH_CHECK(row.numel() == src.size(0), "row length must match src rows");
    TORCH_CHECK(col.numel() == src.size(0), "col length must match src rows");

    row = row.contiguous();
    col = col.contiguous();
    src = src.contiguous();

    int64_t NNZ = row.size(0);
    int64_t F   = src.size(1);

    // allocate output [N, F]
    auto out = torch::zeros({N, F}, src.options());
    if (NNZ == 0 || F == 0 || N == 0) {
        return out;
    }

    const int threads = THREADS_PER_BLOCK;
    const int blocks  = (NNZ * F + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "spmm_cuda", ([&] {
        spmm_cuda_kernel<scalar_t><<<blocks, threads>>>(
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            src.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            NNZ,
            N,
            F
        );
    }));

    return out;
}

// bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_cuda", &spmm_cuda, "SPMM CUDA kernel");
}
