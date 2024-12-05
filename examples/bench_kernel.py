import torch
import time
import argparse
import linear_bf16_cuda
import linear_fp16_cuda
import gptq_cuda
import torch.nn.functional as F
from loguru import logger
from quant import Quantizer, quantize
from module import GPTQLinear


def benchmark(f, warmup=1, iter=10):
    for i in range(warmup):
        out = f()
    torch.cuda.synchronize()
    tick = time.time()
    for i in range(iter):
        out = f()
    torch.cuda.synchronize()
    if isinstance(out, tuple):
        res = out[1] / iter
    else:
        res = (time.time() - tick) / iter
    time.sleep(1.0)
    return res


def initialize_tensors(input_dim, hidden_dim, output_dim, dtype, seed=0):
    torch.manual_seed(seed)
    torch_input = torch.randn(input_dim, hidden_dim, dtype=dtype, device="cuda")
    torch_weight = torch.randn(output_dim, hidden_dim, dtype=dtype, device="cuda")
    torch_bias = torch.randn(output_dim, dtype=dtype, device="cuda")
    custom_output = torch.empty((input_dim, output_dim), dtype=dtype, device="cuda")
    baseline_output = torch.empty((input_dim, output_dim), dtype=dtype, device="cuda")
    return torch_input, torch_weight, torch_bias, custom_output, baseline_output


def run(kernel, input, weight, bias, output):
    if kernel == "w3a16_gptq_cuda":
        output = run_gptq_cuda(input, weight, bias, output)
    elif kernel == "fp16_cuda":
        run_fp16_cuda(input, weight, bias, output)
    elif kernel == "bf16_cuda":
        run_bf16_cuda(input, weight, bias, output)
    elif kernel == "torch":
        output = run_torch_linear(input, weight, bias)
    return output


def run_gptq_cuda(torch_input, torch_weight, torch_bias, custom_output, faster=True):

    DEV = torch.device("cuda:0")
    layer = torch.nn.Linear(torch_weight.size(1), torch_weight.size(0), bias=True)
    layer.weight.data = torch_weight
    if torch_bias is not None:
        layer.bias.data = torch_bias.float()

    quantizer = Quantizer()
    quantizer.configure(3, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = quantize(
        layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
    )
    qlayer = GPTQLinear(layer.in_features, layer.out_features)
    qlayer.pack(layer.cpu(), quantizer.scale.cpu(), quantizer.zero.cpu())
    qlayer = qlayer.to(DEV)
    qlayer.faster = True

    s_t = time.time()
    custom_output = qlayer(torch_input.half())
    e_t = time.time()

    return (custom_output, e_t - s_t)


def run_fp16_cuda(torch_input, torch_weight, torch_bias, custom_output):
    linear_fp16_cuda.linear_forward_fp16(
        torch_input, torch_weight, torch_bias, custom_output
    )


def run_bf16_cuda(torch_input, torch_weight, torch_bias, custom_output):
    linear_bf16_cuda.linear_forward_bf16(
        torch_input, torch_weight, torch_bias, custom_output
    )


def run_torch_linear(torch_input, torch_weight, torch_bias):
    torch_linear = torch.nn.Linear(
        torch_weight.size(1), torch_weight.size(0), bias=True
    ).cuda()
    torch_linear.weight.data = torch_weight
    if torch_linear.bias is not None:
        torch_linear.bias.data = torch_bias
    torch_output = torch_linear(torch_input)
    return torch_output


def compare_outputs(custom_output, baseline_output, k, bk):

    logger.info(f"Custom Linear ({k}) Output:")
    logger.info(custom_output.shape)
    logger.info(f"Baseline Linear ({bk}) Output:")
    logger.info(baseline_output.shape)

    diff = torch.abs(custom_output - baseline_output)
    max_diff = torch.max(diff)
    logger.info(f"Maximum absolute difference between {k} and {bk}: {max_diff.item()}")

    cosine_sim = F.cosine_similarity(
        custom_output.flatten(1), baseline_output.flatten(1), dim=1
    )
    mean_cosine_sim = torch.mean(cosine_sim).item()
    logger.info(f"Mean Cosine Similarity between {k} and {bk}: {mean_cosine_sim}")


def main():

    KERNEL_CHOICES = [
        "w3a16_gptq_cuda",
        "w4a16_marlin_cuda",
        "fp16_cuda",
        "bf16_cuda",
        "torch",
    ]

    parser = argparse.ArgumentParser(description="Benchmark different kernels")

    # [input_dim, hidden_dim] * [output_dim, hidden_dim].T = [input_dim, output_dim]
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1,
        help="Input dim of input tensor",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128 * 4,
        help="Hidden dim of input tensor or weight tensor",
    )

    parser.add_argument(
        "--output_dim",
        type=int,
        default=128,
        help="Output dim of output tensor or weight tensor",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="Dtype of input tensor or weight tensor",
    )

    parser.add_argument(
        "-K",
        "--kernel",
        type=str,
        choices=KERNEL_CHOICES,
        required=True,
        help="Choose the kernel to run",
    )

    parser.add_argument(
        "-BK",
        "--baseline_kernel",
        type=str,
        choices=KERNEL_CHOICES,
        default="torch",
        help="Choose the baseline kernel to compare (default: 'fp16_torch')",
    )

    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iter", type=int, default=10, help="Number of iterations for benchmarking"
    )

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float

    torch_input, torch_weight, torch_bias, custom_output, baseline_output = (
        initialize_tensors(args.input_dim, args.hidden_dim, args.output_dim, dtype)
    )

    custom_output = run(
        args.kernel, torch_input, torch_weight, torch_bias, custom_output
    )
    if isinstance(custom_output, tuple):
        custom_output = custom_output[0]

    baseline_output = run(
        args.baseline_kernel, torch_input, torch_weight, torch_bias, baseline_output
    )

    if args.kernel == "torch":
        k = custom_output.dtype
    else:
        k = args.kernel
    if args.baseline_kernel == "torch":
        bk = baseline_output.dtype
    else:
        bk = args.baseline_kernel

    compare_outputs(custom_output, baseline_output, k, bk)

    custom_kernel = lambda: run(
        args.kernel, torch_input, torch_weight, torch_bias, custom_output
    )
    baseline_kernel = lambda: run(
        args.baseline_kernel, torch_input, torch_weight, torch_bias, baseline_output
    )
    custom_kernel_time = benchmark(custom_kernel, warmup=args.warmup, iter=args.iter)
    baseline_kernel_time = benchmark(
        baseline_kernel, warmup=args.warmup, iter=args.iter
    )

    logger.info(f"Time taken for {k} kernel: {custom_kernel_time:.6f} seconds")
    logger.info(
        f"Time taken for baseline {bk} kernel: {baseline_kernel_time:.6f} seconds"
    )
    if baseline_kernel_time > 0:
        speedup_percentage = (
            (baseline_kernel_time - custom_kernel_time) / baseline_kernel_time
        ) * 100
        logger.info(f"{k} kernel is {speedup_percentage:.2f}% faster than {bk} kernel.")
    else:
        logger.info("Baseline kernel time is zero, cannot calculate speedup.")


if __name__ == "__main__":
    main()
