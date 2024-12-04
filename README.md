# quant_horizon

`quant_horizon` is a benchmarking framework designed to evaluate the performance of different CUDA kernels and compare them with baseline implementations. This framework supports different quantization schemes, including GPTQ (w3a16), FP16, and BF16.

## Prerequisites

To run the benchmark, you need to have the following installed:

- PyTorch (with CUDA support)
- CUDA Toolkit
- Dependencies like `linear_bf16_cuda`, `linear_fp16_cuda`, `gptq_cuda` (assumed to be custom CUDA extensions)

Make sure to install the necessary dependencies using:

```bash
python setup.py install
```

## Usage

### Benchmark GPTQ (w3a16) Kernel
```bash
python examples/bench_kernel.py -K w3a16_gptq_cuda --dtype fp16 -BK torch --input_dim 1 --hidden_dim 512 --output_dim 128
```

### Benchmark FP16 CUDA Kernel
```bash
python examples/bench_kernel.py -K fp16_cuda --dtype fp16 -BK torch --input_dim 1 --hidden_dim 512 --output_dim 128

```

### Benchmark BF16 CUDA Kernel
```bash
python examples/bench_kernel.py -K bf16_cuda --dtype bf16 -BK torch --input_dim 100 --hidden_dim 100 --output_dim 100
```