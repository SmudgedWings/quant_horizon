#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "common/linear_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_wf6af16_linear", &cuda_wf6af16_linear, "DeepSpeed Wf6Af16 linear in CUDA");
    m.def(
        "preprocess_weight", &preprocess_weight, "preprocess the FP16 weight to be 2bit and 4 bit");
}
