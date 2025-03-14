#!/bin/sh

CUDA_VISIBLE_DEVICES=1,2,3 python finetune_dense_lm.py \
    --model_name_or_path /home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/no_finetuning \
    --config_name meta-llama/Llama-3.2-1B \
    --dataset_name tatsu-lab/alpaca \
    --num_train_epochs 4 \
    --block_size 2048 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --loss_type KLDiv \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --use_paser \
    --num_clusters 10 \
    --max_selected_data 30000 \
    --output_dir /home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_KLDiv_PASER

    # CUDA_VISIBLE_DEVICES=1,2 python finetune_dense_lm.py \
    # --model_name_or_path /home/howonlee/HighSparsityPruning/llm_weights/llama2-7B/sparsegpt_0.5/no_finetuning \
    # --config_name meta-llama/Llama-2-7b-hf \
    # --dataset_name tatsu-lab/alpaca \
    # --num_train_epochs 4 \
    # --block_size 2048 \
    # --torch_dtype float16 \
    # --per_device_train_batch_size 1 \
    # --per_device_eval_batch_size 1 \
    # --do_train \
    # --do_eval \
    # --loss_type KLDiv \
    # --learning_rate 1e-4 \
    # --overwrite_output_dir \
    # --use_paser \
    # --num_clusters 10 \
    # --max_selected_data 20000 \
    # --output_dir /home/howonlee/HighSparsityPruning/llm_weights/llama2-7B/sparsegpt_0.5/dense_ft_fp16_1e-4_alpaca_KLDiv_PASER