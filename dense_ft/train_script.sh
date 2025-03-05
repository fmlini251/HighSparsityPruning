#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python finetune_dense_lm.py \
    --model_name_or_path /home/howonlee/HighSparsityPruning/llm_weights/sparsegpt_0.5_llama3.2-1B \
    --config_name meta-llama/Llama-3.2-1B \
    --dataset_name tatsu-lab/alpaca \
    --num_train_epochs 4 \
    --block_size 2048 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir /home/howonlee/HighSparsityPruning/llm_weights/sparsegpt_0.5_llama3.2-1B_dense_ft