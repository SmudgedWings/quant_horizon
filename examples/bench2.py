import torch
from utils.registry_factory import BENCH_REGISTRY
from kernels import *


if __name__ == "__main__":
    A_shape = (16, 4096)
    B_shape = (4096, 4096)
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[1], B_shape[0], dtype=torch.float16, device="cuda")

    init_params = {
        "default": {
            "A_shape": A_shape,
            "B_shape": B_shape,
            "A_data": A_data,
            "B_data": B_data,
        },
        "torch_linear": {},
        # "marlin_quant": {
        #     "groupsize": -1,
        #     "thread_k": 64,
        #     "thread_n": 256,
        #     "sms": -1,
        #     "max_par": 16,
        # },
        "marlin_quant": [
            (
                "perchannel",
                {
                    "groupsize": -1,
                    "thread_k": 64,
                    "thread_n": 256,
                    "sms": -1,
                    "max_par": 16,
                },
            ),
            (
                "pergroup_g128",
                {
                    "groupsize": 128,
                    "thread_k": 64,
                    "thread_n": 256,
                    "sms": -1,
                    "max_par": 16,
                },
            ),
        ],
    }

    BENCH_REGISTRY.benchmark_all(init_params)

    BENCH_REGISTRY.show_all_results()
