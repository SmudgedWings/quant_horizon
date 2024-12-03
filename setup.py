from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='quant_horizon',
    ext_modules=[
        CUDAExtension(
            name='linear_fp16_cuda',
            sources=['float/fp16.cu'],
        ),
        CUDAExtension(
            name='linear_bf16_cuda',
            sources=['float/bf16.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
