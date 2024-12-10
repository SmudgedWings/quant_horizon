import numpy as np
import torch
import torch.nn as nn
import gptq_cuda
from ...quant import IntegerQuantizer
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY


def pack(weight, scales, zeros):
    zeros = zeros * scales
    scales = scales.clone()
    intweight = torch.round((weight + zeros) / scales).to(
        torch.int
    )
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(np.uint32)
    qweight = np.zeros(
        (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
    )
    i = 0
    row = 0
    while row < qweight.shape[0]:
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i))
        i += 10
        qweight[row] |= intweight[i] << 30
        row += 1
        qweight[row] |= (intweight[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 1)
        i += 10
        qweight[row] |= intweight[i] << 31
        row += 1
        qweight[row] |= (intweight[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 2)
        i += 10
        row += 1
    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)
    return qweight.cuda(), scales.cuda().float(), zeros.cuda().float()


def init_gptq_quant(A_shape, B_shape, A_data, B_data):
    assert A_shape[-2] == 1, "Only supports a single token currently."
    quantizer = IntegerQuantizer(bit=3, symmetric=False, granularity="per_channel")
    _, scales, zeros, _, _ = quantizer.get_tensor_qparams(B_data.t())
    weight = quantizer.fake_quant_weight_dynamic(B_data.t())
    qweight, scales, zeros = pack(weight.cpu(), scales.float().cpu(), zeros.float().cpu())
    Y_data = torch.zeros((1, B_shape[1]), device=A_data.device, dtype=A_data.dtype).float()
    return {"A_data": A_data, "B_data_quant": qweight, "Y_data": Y_data, "scales": scales, "zeros": zeros}


def run_gptq_quant(A_data, B_data_quant, Y_data, scales, zeros):
    gptq_cuda.vecquant3matmul_faster(
        A_data,
        B_data_quant,
        Y_data,
        scales,
        zeros,
    )


def get_gptq_quant_res(A_data, B_data_quant, Y_data, scales, zeros):
    gptq_cuda.vecquant3matmul_faster(
        A_data,
        B_data_quant,
        Y_data,
        scales,
        zeros,
    )
    return Y_data


SPEED_REGISTRY.register("gptq_quant", run_gptq_quant, init_gptq_quant)
ACC_REGISTRY.register("gptq_quant", get_gptq_quant_res, init_gptq_quant)
