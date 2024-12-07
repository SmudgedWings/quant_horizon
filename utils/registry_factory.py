import time
import gc
import torch
from tabulate import tabulate
from collections import defaultdict


class BenchRegister(dict):
    def __init__(self, *args, **kwargs):
        super(BenchRegister, self).__init__(*args, **kwargs)
        self._dict = {}
        self._results = defaultdict(dict)
        self._headers = ["Kernel", "Latency/s"]
        self.warmup_iter = kwargs.get("warmup_iter", 10)
        self.bench_iter = kwargs.get("bench_iter", 100)

    def register(self, name, kernel_func, init_func):
        self._dict[name] = {"kernel_func": kernel_func, "init_func": init_func}

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
    def benchmark_func(self, name, kernel_init_params, kernel_tag):
        kernel = self[name]
        prepare_params = kernel["init_func"](**kernel_init_params)

        kernel_func = kernel["kernel_func"]
        for _ in range(self.warmup_iter):
            kernel_func(**prepare_params)

        torch.cuda.synchronize()
        tick = time.time()
        for _ in range(self.bench_iter):
            kernel_func(**prepare_params)
        torch.cuda.synchronize()
        result = (time.time() - tick) / self.bench_iter
        self._results[kernel_tag] = {"Latency/s": result}

        del prepare_params
        gc.collect()
        torch.cuda.empty_cache()

        time.sleep(1.0)

    @torch.no_grad()
    def benchmark(self, name, init_params):
        if not isinstance(init_params[name], list):
            init_params[name] = [(None, init_params[name])]
        for tag, init_param in init_params[name]:
            kernel_init_params = init_params["default"] | init_param
            kernel_tag = name + " + " + tag if tag else name
            self.benchmark_func(name, kernel_init_params, kernel_tag)

    @torch.no_grad()
    def benchmark_all(self, init_params):
        for name in self.keys():
            print(f"{name} bench start.")
            self.benchmark(name, init_params.copy())
            print(f"{name} bench end.")
            print()


BENCH_REGISTRY = BenchRegister()
