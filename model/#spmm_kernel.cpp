#include <torch/extension.h>
#include <vector>

torch::Tensor spmm_cuda(torch::Tensor feat,
                        torch::Tensor rowptr,
                        torch::Tensor colidx);

TORCH_LIBRARY(spmm_lib, m) {
    m.def("spmm", &spmm_cuda);
}
