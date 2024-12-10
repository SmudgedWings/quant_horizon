import torch
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY


def init_torch_linear(A_shape, B_shape, A_data, B_data):
    return {"A_data": A_data, "B_data": B_data}


def run_torch_linear(A_data, B_data):
    torch.matmul(A_data, B_data)


def get_torch_linear_res(A_data, B_data):
    Y_data = torch.matmul(A_data, B_data)
    return Y_data.to(A_data.dtype)


SPEED_REGISTRY.register("torch_linear", run_torch_linear, init_torch_linear)
ACC_REGISTRY.register("torch_linear", get_torch_linear_res, init_torch_linear)


if __name__ == "__main__":
    A_shape = (16, 4096)
    B_shape = (4096, 11008)
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
    }

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()
