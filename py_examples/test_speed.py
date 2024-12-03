import torch
import linear_bf16_cuda
import time

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

t1 = time.time()
torch.cuda.synchronize()
for k in range(20):
    linear_bf16_cuda.linear_forward_bf16(
        torch_input, torch_weight, torch_bias, custom_output
    )
torch.cuda.synchronize()
t2 = time.time()
print(t2 - t1)


torch_linear = torch.nn.Linear(input_size, output_size, bias=True).cuda().half()
torch_linear.weight.data = torch_weight
if torch_linear.bias is not None:
    torch_linear.bias.data = torch_bias


t1 = time.time()
torch.cuda.synchronize()
torch_output = torch_linear(torch_input)
for k in range(20):
    torch_output = torch_linear(torch_input)
torch.cuda.synchronize()
t2 = time.time()
print(t2 - t1)
