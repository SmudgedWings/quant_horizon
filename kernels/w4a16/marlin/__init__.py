# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
from utils.registry_factory import BENCH_REGISTRY


import marlin_cuda


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


# Precompute permutations for Marlin weight and scale shuffling


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


def pack(k, n, s, weight, groupsize=-1):
    tile = 16
    maxq = 2**4 - 1

    w = weight.t()
    if groupsize != k:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
        s = s.reshape((1, -1))
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    if groupsize != k:
        w = w.reshape((groupsize, -1, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((k, n)).contiguous()
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s = s.reshape((-1, n)).contiguous()
    w = w.reshape((k // tile, tile, n // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((k // tile, n * tile))
    res = w
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(w.device)

    return q.to(weight.device), s.to(s.device)


def gen_quant4(m, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((m, n), dtype=torch.half).cuda()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    w = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w

        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    if groupsize == -1:
        groupsize = m
    q, s = pack(m, n, s, w.t(), groupsize)
    return q, s


def init_marlin_quant(
    A_shape, B_shape, A_data, B_data, groupsize, thread_k, thread_n, sms, max_par
):
    assert A_shape == A_data.shape
    assert B_shape == B_data.shape
    B_data_quant, scale = gen_quant4(B_shape[0], B_shape[1], groupsize=groupsize)
    Y_data = torch.zeros((A_shape[0], B_shape[1]), dtype=torch.half).cuda()
    workspace = torch.zeros(B_shape[1] // 128 * 16).cuda()
    return {
        "A_data": A_data,
        "B_data_quant": B_data_quant,
        "scale": scale,
        "Y_data": Y_data,
        "workspace": workspace,
        "thread_k": thread_k,
        "thread_n": thread_n,
        "sms": sms,
        "max_par": max_par,
    }


def run_marlin_quant(
    A_data, B_data_quant, scale, Y_data, workspace, thread_k, thread_n, sms, max_par
):
    mul(
        A_data,
        B_data_quant,
        Y_data,
        scale,
        workspace,
        thread_k=thread_k,
        thread_n=thread_n,
        sms=sms,
        max_par=max_par,
    )


BENCH_REGISTRY.register("marlin_quant", run_marlin_quant, init_marlin_quant)


if __name__ == "__main__":
    A_shape = (16, 4096)
    B_shape = (4096, 11008)
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
        # "marlin_quant": {
        #     "groupsize": -1,
        #     "thread_k": 64,
        #     "thread_n": 256,
        #     "sms": -1,
        #     "max_par": 16,
        # },
        "marlin_quant": [
            (
                "perchannel",
                {
                    "groupsize": -1,
                    "thread_k": -1,
                    "thread_n": -1,
                    "sms": 108, # A100=108, A10=72, 3090=82, A6000=84
                    "max_par": 16,
                },
            ),
            (
                "pergroup_g128",
                {
                    "groupsize": 128,
                    "thread_k": -1,
                    "thread_n": -1,
                    "sms": 108, # A100=108, A10=72, 3090=82, A6000=84
                    "max_par": 16,
                },
            ),
        ],
    }

    BENCH_REGISTRY.benchmark_all(init_params)

    BENCH_REGISTRY.show_all_results()
