from loguru import logger
from transformers import AutoConfig


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
