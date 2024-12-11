# quant_horizon

`quant_horizon` is a benchmarking framework designed to evaluate the performance of different GPU kernels.

## Prerequisites

To run the benchmark, you need to have the following installed:

- PyTorch (with CUDA support)
- CUDA Toolkit

We also provide some basic docker images:
```bash
# docker-hub python3.11 torch2.5.1 cuda124
docker pull llmcompression/llmc:pure-24112502-cu124
# docker-hub python3.11 torch2.5.1 cuda121
docker pull llmcompression/llmc:pure-24112502-cu121
# aliyun-hub python3.11 torch2.5.1 cuda124
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-24112502-cu124
# aliyun-hub python3.11 torch2.5.1 cuda121
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-24112502-cu121

# Then create a container
docker run --gpus all -itd --ipc=host --name [name]  -v [path]:[path] --entrypoint /bin/bash [image_id]
```


Make sure to install the necessary dependencies using:

```bash
cd quant_horizon
pip install -v -e .
```

## Usage

### Benchmark a single shape
```bash
cd examples
python bench_single_shape.py
```

### Benchmark all shapes in the transformer model
```bash
cd examples
# You just need to put the config.json into the model_path folder.
python bench_model_shape.py --model [model_path] --tp 1 --bs 1 --seqlen 2048
```
