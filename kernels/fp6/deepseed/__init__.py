import numpy as np
import torch
import torch.nn as nn
import deepseed_cuda
from ...quant import FloatQuantizer
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
import torch
from torch import nn


def get_workspace(out_channels, tokens, in_channels, split_k, dtype, device):
    """
    Allocate workspace for the kernel. The workspace is used to store the intermediate results of the matmul before
    split-K. The split-K size is determined by the size of the matmul.
    """
    workspace = torch.empty((split_k, out_channels, tokens), dtype=dtype, device=device)
    return workspace


def init_deepseed_quant(A_shape, B_shape, A_data, B_data, split_k):
    quantizer = FloatQuantizer(
        bit="e3m2", symmetric=True, granularity="per_channel", use_qtorch=True
    )
    _, scales, _, _, _ = quantizer.get_tensor_qparams(B_data.t())
    quantized_fake_fp6 = quantizer.quant(B_data.t(), scales, 0, None, None)
    weights_2bit, weights_4bit = deepseed_cuda.preprocess_weight(
        quantized_fake_fp6.half().contiguous().cpu()
    )

    in_channels = B_shape[0]
    out_channels = B_shape[1]
    tokens = A_shape[-2]
    dtype = torch.float
    device = A_data.device
    workspace = get_workspace(out_channels, tokens, in_channels, split_k, dtype, device)
    output = torch.empty((tokens, out_channels), dtype=torch.half, device=device)

    return {
        "output": output,
        "A_data": A_data,
        "weights_2bit": weights_2bit.cuda(),
        "weights_4bit": weights_4bit.cuda(),
        "scales": scales,
        "workspace": workspace,
        "out_channels": out_channels,
        "tokens": tokens,
        "in_channels": in_channels,
        "split_k": split_k,
    }


def run_deepseed_quant(
    output,
    A_data,
    weights_2bit,
    weights_4bit,
    scales,
    workspace,
    out_channels,
    tokens,
    in_channels,
    split_k,
):
    deepseed_cuda.cuda_wf6af16_linear(
        output,
        A_data,
        weights_2bit,
        weights_4bit,
        scales,
        workspace,
        out_channels,
        tokens,
        in_channels,
        split_k,
    )


def get_deepseed_quant_res(
    output,
    A_data,
    weights_2bit,
    weights_4bit,
    scales,
    workspace,
    out_channels,
    tokens,
    in_channels,
    split_k,
):
    deepseed_cuda.cuda_wf6af16_linear(
        output,
        A_data,
        weights_2bit,
        weights_4bit,
        scales,
        workspace,
        out_channels,
        tokens,
        in_channels,
        split_k,
    )
    return output


SPEED_REGISTRY.register("deepseed_quant", run_deepseed_quant, init_deepseed_quant)
ACC_REGISTRY.register("deepseed_quant", get_deepseed_quant_res, init_deepseed_quant)
