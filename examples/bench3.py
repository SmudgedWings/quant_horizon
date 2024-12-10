import torch
from loguru import logger
from transformers import AutoConfig
from kernels import *
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY


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


def bench_shape(A_shape, B_shape):
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

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--seqlen", type=int)
    args = parser.parse_args()

    linear_size = get_linear_size(args.model, args.tp)
    logger.info(f"linear_size : {linear_size}")

    mm_size = get_mm_size(linear_size, bs=args.bs, seqlen=args.seqlen)
    logger.info(f"mm_size : {mm_size}")

    for layer in mm_size:
        A_shape, B_shape = mm_size[layer]["prefill"]
        print("*" * 60)
        print(f"{layer}-prefill: {A_shape} x {B_shape}")
        bench_shape(A_shape, B_shape)
        print()

        A_shape, B_shape = mm_size[layer]["decode"]
        print("*" * 60)
        print(f"{layer}-decode: {A_shape} x {B_shape}")
        bench_shape(A_shape, B_shape)
        print()
