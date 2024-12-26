# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
import torch
import qserve_backend_qgemm_w4a8_per_chn
import qserve_backend_qgemm_w4a8_per_group
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY


def init_w4a8g128_qoq_quant(A_shape, B_shape, A_data, B_data):
    qweight = torch.rand(B_shape[0], B_shape[1] // 2).to(torch.int8).cuda()
    s1_scale = torch.rand(B_shape[1]).to(torch.float16).cuda()
    s2_scale = torch.rand(B_shape[0] // 128, B_shape[1]).to(torch.int8).cuda()
    zeros = torch.rand(B_shape[0] // 128, B_shape[1]).to(torch.int8).cuda()

    qinput = torch.rand(A_shape[0], A_shape[1]).to(torch.int8).cuda()
    quantized_scale_buffer = torch.rand(A_shape[0]).to(torch.float16).cuda()
    qkv_proj_act_buffer = torch.rand(A_shape[0], A_shape[1]).to(torch.float16).cuda()
    return {
        "qinput": qinput,
        "qweight": qweight,
        "zeros": zeros,
        "s2_scale": s2_scale,
        "s1_scale": s1_scale,
        "quantized_scale_buffer": quantized_scale_buffer,
        "qkv_proj_act_buffer": qkv_proj_act_buffer,
    }


def run_w4a8g128_qoq_quant(
    qinput, qweight, zeros, s2_scale, s1_scale, quantized_scale_buffer, qkv_proj_act_buffer
):
    qserve_backend_qgemm_w4a8_per_group.gemm_forward_cuda(
        qinput,
        qweight,
        zeros,
        s2_scale,
        s1_scale,
        quantized_scale_buffer,
        qkv_proj_act_buffer,
    )


def init_w4a8_qoq_quant(A_shape, B_shape, A_data, B_data):
    qweight = torch.rand(B_shape[0], B_shape[1] // 2).to(torch.int8).cuda()
    s1_scales = torch.rand(B_shape[1]).to(torch.float16).cuda()
    s1_szeros = torch.rand(B_shape[1]).to(torch.float16).cuda()

    qinput = torch.rand(A_shape[0], A_shape[1]).to(torch.int8).cuda()
    quantized_scale_buffer = torch.rand(A_shape[0]).to(torch.float16).cuda()
    quantized_sum_buffer = torch.rand(5).to(torch.float16).cuda()
    qkv_proj_act_buffer = torch.rand(A_shape[0], A_shape[1]).to(torch.float16).cuda()
    return {
        "qinput": qinput,
        "qweight": qweight,
        "s1_scales": s1_scales,
        "quantized_scale_buffer": quantized_scale_buffer,
        "s1_szeros": s1_szeros,
        "quantized_sum_buffer": quantized_sum_buffer,
        "qkv_proj_act_buffer": qkv_proj_act_buffer,
    }


def run_w4a8_qoq_quant(
    qinput, qweight, s1_scales, quantized_scale_buffer, s1_szeros, quantized_sum_buffer, qkv_proj_act_buffer
):
    qserve_backend_qgemm_w4a8_per_chn.gemm_forward_cuda(
        qinput,
        qweight,
        s1_scales,
        quantized_scale_buffer,
        s1_szeros,
        quantized_sum_buffer,
        qkv_proj_act_buffer,
    )


SPEED_REGISTRY.register("w4a8_qoq", run_w4a8_qoq_quant, init_w4a8_qoq_quant)
SPEED_REGISTRY.register("w4a8g128_qoq", run_w4a8g128_qoq_quant, init_w4a8g128_qoq_quant)


if __name__ == "__main__":
    # qweight = torch.rand(4096, 2048).to(torch.int8)
    # s1_scale = torch.rand(4096).to(torch.float16)
    # s2_scale = torch.rand(32, 4096).to(torch.int8)
    # zeros = torch.rand(32, 4096).to(torch.int8)

    # quantized_hidden_states_buffer = torch.rand(5, 4096).to(torch.int8).cuda()
    # quantized_scale_buffer = torch.rand(5).to(torch.float16).cuda()
    # quantized_sum_buffer = torch.rand(5).to(torch.float16).cuda()
    # qkv_proj_act_buffer = torch.rand(5, 4096).to(torch.float16).cuda()

    # qserve_backend_qgemm_w4a8_per_chn.gemm_forward_cuda(
    #     quantized_hidden_states_buffer,
    #     qweight,
    #     zeros,
    #     s2_scale,
    #     s1_scale,
    #     quantized_scale_buffer,
    #     qkv_proj_act_buffer,
    # )


    A_shape = (16, 4096)
    B_shape = (4096, 8192)
    A_data = torch.randn(A_shape[0], A_shape[1], dtype=torch.float16, device="cuda")
    B_data = torch.randn(B_shape[0], B_shape[1], dtype=torch.float16, device="cuda")
    # init_params = {
    #     "default": {
    #         "A_shape": A_shape,
    #         "B_shape": B_shape,
    #         "A_data": A_data,
    #         "B_data": B_data,
    #     },
    #     "torch_matmul": {},
    # }

    p = init_w4a8g128_qoq_quant(A_shape, B_shape, A_data, B_data)
    run_w4a8g128_qoq_quant(**p)

    p = init_w4a8_qoq_quant(A_shape, B_shape, A_data, B_data)
    run_w4a8_qoq_quant(**p)
