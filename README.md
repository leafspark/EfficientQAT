# EfficientQAT

Official PyTorch implementation of [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062)

## News
- [2024/08] Added support for [Mistral-Large-Instruct](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) quantization. W2g64 Mistral-Large-Instruct compressed to 35 GB with only 4% accuracy loss.
- [2024/07] New feature: Transfer EfficientQAT quantized models to `GPTQ v2` and `BitBLAS` formats, loadable through [GPTQModel](https://github.com/ModelCloud/GPTQModel).
- [2024/07] Initial release of EfficientQAT, pushing the limits of uniform (INT) quantization efficiently.

## Contents
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Inference](#inference)
- [Model Transferring](#model-transferring)
- [Inference with Other Formats](#inference-with-other-formats)
- [Citation](#citation)

## Installation

1. Clone the repository:
```
git clone https://github.com/OpenGVLab/EfficientQAT.git
cd EfficientQAT
```

2. Set up the environment:
```
conda create -n efficientqat python=3.11
conda activate efficientqat
pip install -r requirements.txt
```

## Model Zoo

We provide pre-quantized EfficientQAT models. For details, see the [full model table](#) in the original README.

## Training

EfficientQAT involves two training phases: Block-wise training (Block-AP) and end-to-end quantization parameter training (E2E-QP). 

### Block-AP

Modify the `--model` path in the script, then run:

```batch
examples/block_ap/LlamaForCasualLM/w2g64.bat
```

### E2E-QP

Modify the `--quant_model_path` in the script, then run:

For RedPajama dataset:
```batch
examples/e2e_qp/Llama-2-7b/w2g64-redpajama.bat
```

For Alpaca dataset:
```batch
examples/e2e_qp/Llama-2-7b/w2g64-alpaca.bat
```

## Inference

1. Download pre-quantized models:
```
pip install huggingface_hub
huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w2g64 --local-dir ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64
```

2. Evaluate:
```batch
@echo off
set CUDA_VISIBLE_DEVICES=0
python main_block_ap.py ^
--resume_quant ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w2g64 ^
--net Llama-2 ^
--wbits 2 ^
--group_size 64 ^
--output_dir ./output/inference_results/ ^
--eval_ppl ^
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Model Transferring

Install `gptqmodel`:
```
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel
bash install.sh
```

Transfer options:

1. To GPTQ format:
```batch
examples/model_transfer/efficientqat_to_gptq/LlamaForCasualLM.bat
```

2. To BitBLAS format:
```batch
examples/model_transfer/efficientqat_to_bitblas/LlamaForCasualLM.bat
```

3. Convert fp32 to half-precision:
```batch
examples/model_transfer/fp32_to_16/LlamaForCasualLM.bat
```

## Inference with Other Formats

Example for GPTQ or BitBLAS formats:

```python
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

quant_dir = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-GPTQ"
# or "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128-BitBLAS"

tokenizer = AutoTokenizer.from_pretrained(quant_dir, use_fast=True)
model = GPTQModel.from_quantized(quant_dir)

print(tokenizer.decode(model.generate(**tokenizer("Model quantization is", return_tensors="pt").to(model.device))[0]))
```

## Citation

If you find this work useful, please cite:
```
@article{efficientqat,
  title={EfficientQAT: Efficient Quantization-Aware Training for Large Language Models},
  author={Chen, Mengzhao and Shao, Wenqi and Xu, Peng and Wang, Jiahao and Gao, Peng and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2407.11062},
  year={2024}
}
```