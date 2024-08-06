@echo off
set CUDA_VISIBLE_DEVICES=0
python main_block_ap.py ^
--model path/to/Llama-2-7b ^
--output_dir ./output/block_ap_log/Llama-2-7b-w2g64 ^
--net Llama-2 ^
--wbits 2 ^
--group_size 64 ^
--quant_lr 1e-4 ^
--weight_lr 2e-5 ^
--real_quant ^
--eval_ppl ^
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande ^
--save_quant_dir ./output/block_ap_models/Llama-2-7b-w2g64
