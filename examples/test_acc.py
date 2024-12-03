import torch
import linear_bf16_cuda

batch_size = 4
input_size = 8
output_size = 16

dtype = torch.bfloat16

torch.manual_seed(42)


torch_input = torch.randn(batch_size, input_size, dtype=dtype, device="cuda")
torch_weight = torch.randn(output_size, input_size, dtype=dtype, device="cuda")
torch_bias = torch.randn(output_size, dtype=dtype, device="cuda")

custom_output = torch.empty((batch_size, output_size), dtype=dtype, device="cuda")
linear_bf16_cuda.linear_forward_bf16(
    torch_input, torch_weight, torch_bias, custom_output
)

torch_linear = torch.nn.Linear(input_size, output_size, bias=True).cuda().half()
torch_linear.weight.data = torch_weight
if torch_linear.bias is not None:
    torch_linear.bias.data = torch_bias

torch_output = torch_linear(torch_input)

print("Custom CUDA Linear Output:")
print(custom_output)
print("\nTorch Linear Output:")
print(torch_output)

diff = torch.abs(custom_output - torch_output)
max_diff = torch.max(diff)
print(f"\nMaximum absolute difference: {max_diff.item()}")
