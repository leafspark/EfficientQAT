@echo off
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
python -m model_transfer.efficientqat_to_others ^
--model path/to/original/quantized/model ^
--save_dir path/to/new/model ^
--wbits 2 ^
--group_size 64 ^
--test_speed ^
--target_format bitblas
