import torch
import yaml
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
from kernels import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yml")
    args = parser.parse_args()
    with open(args.cfg, "r") as file:
        init_params = yaml.safe_load(file)
    print(init_params)
    print()

    A_shape = (1, 4096)
    B_shape = (4096, 8192)
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[0], B_shape[1], dtype=torch.float16, device="cuda")

    init_params["default"]["A_shape"] = A_shape
    init_params["default"]["B_shape"] = B_shape
    init_params["default"]["A_data"] = A_data
    init_params["default"]["B_data"] = B_data

    print("*" * 60)
    print(f"shape is: {A_shape} x {B_shape}")

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()
