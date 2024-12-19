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
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="marlin_cuda_quant",
            sources=[
                "kernels/w4a16/marlin/marlin_cuda.cpp",
                "kernels/w4a16/marlin/marlin_cuda_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="marlin_cuda_sparse",
            sources=[
                "kernels/w4a16_sparse/marlin/marlin_cuda.cpp",
                "kernels/w4a16_sparse/marlin/marlin_cuda_kernel_nm.cu",
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
        CUDAExtension(
            name="deepseed_cuda",
            sources=[
                "kernels/fp6/deepseed/pybind_deepseed.cpp",
                "kernels/fp6/deepseed/common/linear_kernels_cuda.cu",
                "kernels/fp6/deepseed/common/linear_kernels.cpp",
            ],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="fp8_cuda",
            sources=[
                "kernels/fp8/fp8_cuda_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="qserve_backend_qgemm_w4a8_per_group",
            sources=[
                "kernels/w4a8/qoq/csrc/qgemm/w4a8_per_group/pybind.cpp",
                "kernels/w4a8/qoq/csrc/qgemm/w4a8_per_group/gemm_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-g",
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                    "-std=c++17",
                    "-DENABLE_BF16",
                ],
                "nvcc": extra_compile_args["nvcc"]
                + [
                    "-O2",
                    "-std=c++17",
                    "-DENABLE_BF16",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--threads=8",
                ],
            },
        ),
        CUDAExtension(
            name="qserve_backend_qgemm_w4a8_per_chn",
            sources=[
                "kernels/w4a8/qoq/csrc/qgemm/w4a8_per_chn/pybind.cpp",
                "kernels/w4a8/qoq/csrc/qgemm/w4a8_per_chn/gemm_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-g",
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                    "-std=c++17",
                    "-DENABLE_BF16",
                ],
                "nvcc": extra_compile_args["nvcc"]
                + [
                    "-O2",
                    "-std=c++17",
                    "-DENABLE_BF16",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--threads=8",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
