import torch
import yaml
from loguru import logger
from transformers import AutoConfig
from kernels import *
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
from collections import defaultdict
import argparse


def get_linear_size(model_path, tp):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"model_config : {model_config}")

    architectures = model_config.architectures
    logger.info(f"architectures : {architectures}")

    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    logger.info(f"hidden_size : {hidden_size}")
    logger.info(f"intermediate_size : {intermediate_size}")

    # Y = X * W.t()
    if architectures[0] in ["LlamaForCausalLM"]:
        assert hidden_size % tp == 0
        assert intermediate_size % tp == 0
        q_linear = (hidden_size // tp, hidden_size)  # (out_features, in_features)
        k_linear = (hidden_size // tp, hidden_size)
        v_linear = (hidden_size // tp, hidden_size)
        o_linear = (hidden_size, hidden_size // tp)

        gate_linear = (intermediate_size // tp, hidden_size)
        up_linear = (intermediate_size // tp, hidden_size)
        down_linear = (hidden_size, intermediate_size // tp)

        qkv_linear = (3 * hidden_size // tp, hidden_size)
        kv_linear = (2 * hidden_size // tp, hidden_size)
        gateup_linear = (2 * intermediate_size // tp, hidden_size)

        return_dict = {
            "q_linear": q_linear,
            "k_linear": k_linear,
            "v_linear": v_linear,
            "o_linear": o_linear,
            "gate_linear": gate_linear,
            "up_linear": up_linear,
            "down_linear": down_linear,
            "qkv_linear": qkv_linear,
            "kv_linear": kv_linear,
            "gateup_linear": gateup_linear,
        }

    return return_dict


def get_mm_size(linear_size, bs, seqlen):
    return_dict = {}
    for name in linear_size:
        return_dict[name] = {
            "prefill": (
                (bs * seqlen, linear_size[name][1]),
                (linear_size[name][1], linear_size[name][0]),
            ),
            "decode": (
                (bs * 1, linear_size[name][1]),
                (linear_size[name][1], linear_size[name][0]),
            ),
        }
    return return_dict


def deduplication(mm_size):
    mm_size_compact = defaultdict(list)
    for layer in mm_size:
        for mode in mm_size[layer]:
            mm_size_compact[mm_size[layer][mode]].append(layer + "_" + mode)
    return mm_size_compact


def bench_shape(A_shape, B_shape, init_params):
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[0], B_shape[1], dtype=torch.float16, device="cuda")

    init_params["default"]["A_shape"] = A_shape
    init_params["default"]["B_shape"] = B_shape
    init_params["default"]["A_data"] = A_data
    init_params["default"]["B_data"] = B_data

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yml")
    parser.add_argument("--model", type=str)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--seqlen", type=int)
    args = parser.parse_args()

    with open(args.cfg, "r") as file:
        init_params = yaml.safe_load(file)
    print(init_params)

    linear_size = get_linear_size(args.model, args.tp)
    logger.info(f"linear_size : {linear_size}")

    mm_size = get_mm_size(linear_size, bs=args.bs, seqlen=args.seqlen)
    logger.info(f"mm_size : {mm_size}")

    mm_size_compact = deduplication(mm_size)
    logger.info(f"mm_size_compact : {mm_size_compact}")

    for AB_shape in mm_size_compact:
        A_shape, B_shape = AB_shape
        print("*" * 60)
        print(f"{A_shape} x {B_shape} for {mm_size_compact[AB_shape]}")
        bench_shape(A_shape, B_shape, init_params.copy())
        print()
