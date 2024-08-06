@echo off
set CUDA_VISIBLE_DEVICES=0
python main_block_ap.py ^
--model path/to/Mistral-Large-Instruct-2407 ^
--output_dir ./output/block_ap_log/Mistral-Large-Instruct-2407-w2g64 ^
--net mistral-large ^
--wbits 2 ^
--group_size 64 ^
--quant_lr 3e-5 ^
--weight_lr 2e-6 ^
--train_size 2048 ^
--epochs 3 ^
--eval_ppl ^
--real_quant ^
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande ^
--save_quant_dir ./output/block_ap_models/Mistral-Large-Instruct-2407-w2g64
