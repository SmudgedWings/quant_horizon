import numpy as np
import torch
import torch.nn as nn
import gptq_cuda
from ...quant import BaseQuantizer, IntegerQuantizer
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY


class GPTQLinear(nn.Module):
    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer(
            "qweight", torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(
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
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                gptq_cuda.vecquant3matmul_faster(
                    x, self.qweight, y, self.scales, self.zeros
                )
            else:
                x = x.float()
                gptq_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError("Only supports a single token currently.")


def init_gptq_quant(A_shape, B_shape, A_data, B_data):
    assert A_shape[-2] == 1, "Only supports a single token currently."
    DEV = torch.device("cuda:0")
    layer = torch.nn.Linear(B_shape[0], B_shape[1], bias=False)
    layer.weight.data = B_data.t()
    quantizer = IntegerQuantizer(bit=3, symmetric=False, granularity="per_channel")
    _, scales, zeros, qmax, qmin = quantizer.get_tensor_qparams(layer.weight.data)
    layer.weight.data = quantizer.fake_quant_weight_dynamic(layer.weight.data)

    qlayer = GPTQLinear(layer.in_features, layer.out_features)
    qlayer.pack(layer.cpu(), scales.float().cpu(), zeros.float().cpu())
    qlayer = qlayer.to(DEV)
    qlayer.faster = True
    Y_data = torch.zeros((1, B_shape[1]), device=A_data.device, dtype=A_data.dtype)
    return {"A_data": A_data, "qlayer": qlayer, "Y_data": Y_data}


def run_gptq_quant(A_data, qlayer, Y_data):
    gptq_cuda.vecquant3matmul_faster(
        A_data,
        qlayer.qweight,
        Y_data.float(),
        qlayer.scales.float(),
        qlayer.zeros.float(),
    )


def get_gptq_quant_res(A_data, qlayer, Y_data):
    Y_data = qlayer(A_data.half())
    return Y_data.to(A_data.dtype)


SPEED_REGISTRY.register("gptq_quant", run_gptq_quant, init_gptq_quant)
ACC_REGISTRY.register("gptq_quant", get_gptq_quant_res, init_gptq_quant)
