#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void linear_forward_fp16(const __half* input, const __half* weight, const __half* bias, __half* output,
                                    int input_size, int output_size, int batch_size) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < batch_size && col < output_size) {
        __half sum = __float2half(0.0f);
        for (int i = 0; i < input_size; ++i) {
            sum = __hadd(sum, __hmul(input[row * input_size + i], weight[col * input_size + i]));
        }
        output[row * output_size + col] = bias ? __hadd(sum, bias[col]) : sum;
    }
}

void linear_forward_cuda_fp16(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);

    const dim3 blocks(batch_size);
    const dim3 threads(output_size);

    AT_CUDA_CHECK(cudaGetLastError());
    linear_forward_fp16<<<blocks, threads>>>(
        reinterpret_cast<__half*>(input.data_ptr()),
        reinterpret_cast<__half*>(weight.data_ptr()),
        bias.defined() ? reinterpret_cast<__half*>(bias.data_ptr()) : nullptr,
        reinterpret_cast<__half*>(output.data_ptr()),
        input_size, output_size, batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward_fp16", &linear_forward_cuda_fp16, "Linear forward with fp16 (CUDA)");
}
