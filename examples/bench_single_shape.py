import torch
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
from kernels import *


if __name__ == "__main__":
    A_shape = (1, 4096)
    B_shape = (4096, 8192)
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[0], B_shape[1], dtype=torch.float16, device="cuda")

    init_params = {
        "default": {
            "A_shape": A_shape,
            "B_shape": B_shape,
            "A_data": A_data,
            "B_data": B_data,
        },
        "torch_linear": {},
        "marlin_quant": [
            (
                "perchannel",
                {
                    "groupsize": -1,
                    "thread_k": -1,
                    "thread_n": -1,
                    "sms": 108,
                    "max_par": 16,
                },
            ),
            (
                "pergroup_g128",
                {
                    "groupsize": 128,
                    "thread_k": -1,
                    "thread_n": -1,
                    "sms": 108,
                    "max_par": 16,
                },
            ),
        ],
        "marlin_quant_sparse": [
            (
                "perchannel",
                {
                    "groupsize": -1,
                    "thread_k": -1,
                    "thread_m": -1,
                    "sms": 108,  # A100=108, A10=72, 3090=82, A6000=84
                    "max_par": 16,
                },
            ),
            (
                "pergroup_g128",
                {
                    "groupsize": 128,
                    "thread_k": -1,
                    "thread_m": -1,
                    "sms": 108,  # A100=108, A10=72, 3090=82, A6000=84
                    "max_par": 16,
                },
            ),
        ],
        "gptq_quant": [("w3a16_perchannel", {})],
    }

    print(f"shape is: {A_shape} x {B_shape}")

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()
