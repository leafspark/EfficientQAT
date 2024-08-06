set CUDA_VISIBLE_DEVICES=0
python main_e2e_qp.py ^
    --quant_model_path .\output\block_ap_models\Llama-2-7b-w4g128 ^
    --model_family Llama-2 ^
    --wbits 4 ^
    --group_size 128 ^
    --learning_rate 1e-5 ^
    --dataset alpaca ^
    --dataset_format alpaca ^
    --output_dir .\output\e2e-qp-output\Llama-2-7b-w4g128-alpaca-4096 ^
    --do_train True ^
    --do_mmlu_eval True ^
    --source_max_len 384 ^
    --target_max_len 128 ^
    --per_device_train_batch_size 16 ^
    --per_device_eval_batch_size 4 ^
    --gradient_accumulation_steps 1 ^
    --logging_steps 10 ^
    --save_strategy steps ^
    --evaluation_strategy steps ^
    --max_steps 10000 ^
    --eval_steps 2000 ^
    --eval_dataset_size 16 ^
    --bf16 ^
    --data_seed 42 ^
    --max_grad_norm 0.3 ^
    --group_by_length
