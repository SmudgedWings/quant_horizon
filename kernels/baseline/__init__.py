import torch
from utils.registry_factory import BENCH_REGISTRY


def init_torch_linear(A_shape, B_shape, A_data, B_data):
    assert A_shape == A_data.shape
    assert B_shape == B_data.shape
    torch_linear = torch.nn.Linear(B_shape[1], B_shape[0], bias=False).cuda()
    torch_linear.weight.data = B_data
    return {"A_data": A_data, "torch_linear": torch_linear}


def run_torch_linear(A_data, torch_linear):
    torch_linear(A_data)


BENCH_REGISTRY.register("torch_linear", run_torch_linear, init_torch_linear)


if __name__ == "__main__":

    def init_torch_linear(A_shape, B_shape, A_data, B_data):
        assert A_shape == A_data.shape
        assert B_shape == B_data.shape
        torch_linear = torch.nn.Linear(B_shape[1], B_shape[0], bias=False).cuda()
        torch_linear.weight.data = B_data
        return {"A_data": A_data, "torch_linear": torch_linear}

    def run_torch_linear(A_data, torch_linear):
        torch_linear(A_data)

    BENCH_REGISTRY.register("torch_linear", run_torch_linear, init_torch_linear)

    A_shape = (8192, 4096)
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
    }

    # BENCH_REGISTRY.benchmark("torch_linear", init_params)
    BENCH_REGISTRY.benchmark_all(init_params)

    BENCH_REGISTRY.show_all_results()
