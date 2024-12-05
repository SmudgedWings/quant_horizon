from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


extra_compile_args = {
    "nvcc": [
        "-gencode",
        "arch=compute_80,code=sm_80",  # A100
        "-gencode",
        "arch=compute_86,code=sm_86",  # 3090
        "-gencode",
        "arch=compute_89,code=sm_89",  # 4090
        "-gencode",
        "arch=compute_90,code=sm_90",  # H100
        "--ptxas-options=-v",
        "-lineinfo",
    ]
}

setup(
    name="quant_horizon",
    ext_modules=[
        CUDAExtension(
            name="linear_fp16_cuda",
            sources=["kernels/fp_bf/fp16.cu"],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="linear_bf16_cuda",
            sources=["kernels/fp_bf/bf16.cu"],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="marlin_cuda",
            sources=[
                "kernels/w4a16/marlin/marlin_cuda.cpp",
                "kernels/w4a16/marlin/marlin_cuda_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="gptq_cuda",
            sources=[
                "kernels/w3a16/gptq/quant_cuda.cpp",
                "kernels/w3a16/gptq/quant_cuda_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
