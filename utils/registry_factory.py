import time
import gc
import torch
from tabulate import tabulate


class BenchRegister(dict):
    def __init__(self, *args, **kwargs):
        super(BenchRegister, self).__init__(*args, **kwargs)
        self._dict = {}
        self._results = {}
        self._headers = ["Kernel", "Latency/s"]
        self.warmup_iter = kwargs.get("warmup_iter", 2)
        self.bench_iter = kwargs.get("bench_iter", 10)

    def register(self, name, kernel_func, init_func):
        self._dict[name] = {"kernel_func": kernel_func, "init_func": init_func}
        self._results[name] = {}

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def sort_tabulate(self, data):
        for i, sublist in enumerate(data):
            if sublist[0] == "torch_linear":
                k_sublist = data.pop(i)
                data.insert(0, k_sublist)

    def show_all_results(self):
        print(self._results)
        data = []
        for name in self._results:
            data.append(
                [name] + [self._results[name].get(h, "-") for h in self._headers[1:]]
            )
        self.sort_tabulate(data)
        print(tabulate(data, headers=self._headers, tablefmt="psql", stralign="center"))

    def show_result(self, name):
        print(self._results[name])

    @torch.no_grad()
    def benchmark(self, name, init_params):
        kernel = self[name]
        kernel_init_params = init_params["default"] | init_params[name]
        prepare_params = kernel["init_func"](**kernel_init_params)

        kernel_func = kernel["kernel_func"]
        for _ in range(self.warmup_iter):
            kernel_func(**prepare_params)

        torch.cuda.synchronize()
        tick = time.time()
        for _ in range(self.bench_iter):
            kernel_func(**prepare_params)
        torch.cuda.synchronize()
        result = (time.time() - tick) / self.warmup_iter
        self._results[name] = {"Latency/s": result}

        del prepare_params
        gc.collect()
        torch.cuda.empty_cache()

        time.sleep(1.0)

    @torch.no_grad()
    def benchmark_all(self, init_params):
        for name in self.keys():
            print(f"{name} bench start.")
            self.benchmark(name, init_params)
            print(f"{name} bench end.")
            print()


BENCH_REGISTRY = BenchRegister()


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
