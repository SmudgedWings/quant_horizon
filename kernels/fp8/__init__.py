from typing import Optional, Tuple, Union
import torch
import fp8_cuda
from utils.registry_factory import SPEED_REGISTRY, ACC_REGISTRY
def is_hip() -> bool:
    return torch.version.hip is not None

TORCH_SCALED_MM_SCALE_RESULT = torch.ones(1).cuda() if is_hip() else None

# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic 
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token 
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    # # For rocm, the output fp8 dtype is torch.float_e3m3fnuz
    # out_dtype: torch.dtype = torch.float8_e4m3fnuz if vllm.utils.is_hip() \
    #     else torch.float8_e4m3fn
    # For rocm, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = torch.float8_e4m3fn
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            fp8_cuda.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            fp8_cuda.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert (scale.numel() == 1 or num_token_padding is None)
        fp8_cuda.static_scaled_fp8_quant(output, input, scale)

    return output, scale

def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
   

    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    qinput, x_scale = scaled_fp8_quant(
        input,
        input_scale,
        num_token_padding=17,
        use_per_token_if_dynamic=use_per_token_if_dynamic)

    per_tensor_weights = (weight_scale.numel() == 1)
    per_tensor_activations = (x_scale.numel() == 1)

    if per_tensor_weights and per_tensor_activations:
        # Fused GEMM_DQ
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            scale_result=TORCH_SCALED_MM_SCALE_RESULT,
            bias=bias)
        # Since in torch 2.5, scaled_mm only returns single value
        # This should be removed when vllm-nvidia also moves to 2.5
        if is_hip():
            return torch.narrow(output, 0, 0, input.shape[0])
        return torch.narrow(output[0], 0, 0, input.shape[0])

    else:
        # Fallback for channelwise case, where we use unfused DQ
        # due to limitations with scaled_mm

        # Symmetric quantized GEMM by definition computes the following:
        #   C = (s_x * X) (s_w * W) + bias
        # This is equivalent to dequantizing the weights and activations
        # before applying a GEMM.
        #
        # In order to compute quantized operands, a quantized kernel
        # will rewrite the above like so:
        #   C = s_w * s_x * (X * W) + bias
        #
        # For the scaled_mm fallback case, we break this down, since it
        # does not support s_w being a vector.

        # GEMM
        # This computes C = (X * W).
        # Output in fp32 to allow subsequent ops to happen in-place
        output, _ = torch._scaled_mm(qinput,
                                        weight,
                                        out_dtype=torch.float32)
        # Unpad (undo num_token_padding)
        output = torch.narrow(output, 0, 0, input.shape[0])
        x_scale = torch.narrow(x_scale, 0, 0, input.shape[0])

        # DQ
        # C = sw * sx * (X * W) + bias
        output = output * x_scale * weight_scale.t()
        if bias is not None:
            output = output + bias
        return output.to(dtype=input.dtype)

def init_fp8_static(A_shape, B_shape, A_data, B_data):
    A_data_fp8, scale_A = scaled_fp8_quant(A_data)
    B_data_fp8, scale_B = scaled_fp8_quant(B_data)
    B_data_fp8 = B_data_fp8.t().contiguous().t()
    return {
        "A_data_fp8":A_data_fp8, "scale_A":scale_A,
        "B_data_fp8":B_data_fp8, "scale_B":scale_B
    }

def run_fp8_static(A_data_fp8, scale_A, B_data_fp8, scale_B):
    torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a = scale_A,
        scale_b = scale_B,
        scale_result=TORCH_SCALED_MM_SCALE_RESULT
    )
    
def get_fp8_static_res(A_data_fp8, scale_A, B_data_fp8, scale_B):
    C_fp8 = torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a = scale_A,
        scale_b = scale_B,
        scale_result=TORCH_SCALED_MM_SCALE_RESULT
    )
    return C_fp8.to(torch.float16)

def init_fp8_dynamic(A_shape, B_shape, A_data, B_data):
    B_data_fp8, scale_B = scaled_fp8_quant(B_data)
    B_data_fp8 = B_data_fp8.t().contiguous().t()
    return {"A_data":A_data, "B_data_fp8":B_data_fp8, "scale_B":scale_B}

def run_fp8_dynamic(A_data, B_data_fp8, scale_B):
    A_data_fp8, scale_A = scaled_fp8_quant(A_data)
    torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a = scale_A,
        scale_b = scale_B,
        scale_result=TORCH_SCALED_MM_SCALE_RESULT
    )
    
def get_fp8_dynamic_res(A_data, B_data_fp8, scale_B):
    A_data_fp8, scale_A = scaled_fp8_quant(A_data)
    C_fp8 = torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a = scale_A,
        scale_b = scale_B,
        scale_result=TORCH_SCALED_MM_SCALE_RESULT
    )
    return C_fp8.to(torch.float16)

def run_fp8_dynamic_per_token(A_data, B_data_fp8, scale_B):
    A_data_fp8, scale_A = scaled_fp8_quant(A_data,use_per_token_if_dynamic=True)
    output = torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a=torch.tensor(1.0, device=scale_A.device),
        scale_b=torch.tensor(1.0, device=scale_B.device),
        out_dtype=torch.float32
    )
    output = output * scale_A * scale_B.t()
    
def get_fp8_dynamic_per_token_res(A_data, B_data_fp8, scale_B):
    A_data_fp8, scale_A = scaled_fp8_quant(A_data,use_per_token_if_dynamic=True)
    output = torch._scaled_mm(
        A_data_fp8,
        B_data_fp8,
        scale_a=torch.tensor(1.0, device=scale_A.device),
        scale_b=torch.tensor(1.0, device=scale_B.device),
        out_dtype=torch.float32
    )
    output = output * scale_A * scale_B.t()
    return output.to(torch.float16)

SPEED_REGISTRY.register("fp8_static", run_fp8_static, init_fp8_static)
ACC_REGISTRY.register("fp8_static", get_fp8_static_res, init_fp8_static)

SPEED_REGISTRY.register("fp8_dynamic", run_fp8_dynamic, init_fp8_dynamic)
ACC_REGISTRY.register("fp8_dynamic", get_fp8_dynamic_res, init_fp8_dynamic)

SPEED_REGISTRY.register("fp8_dynamic_per_token", run_fp8_dynamic_per_token, init_fp8_dynamic)
ACC_REGISTRY.register("fp8_dynamic_per_token", get_fp8_dynamic_per_token_res, init_fp8_dynamic)

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
        "fp8_static": {},
        "fp8_dynamic": {},
        "fp8_dynamic_per_token": {},
    }

    SPEED_REGISTRY.benchmark_all(init_params)
    SPEED_REGISTRY.show_all_results()

    ACC_REGISTRY.benchmark_all(init_params)
    ACC_REGISTRY.show_all_results()