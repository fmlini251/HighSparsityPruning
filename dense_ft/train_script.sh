#!/bin/sh

# python finetune_dense_lm.py \
#     --model_name_or_path /home/howon010402/HighSparsityPruning/llm_weights/llama-3.1-8B/sparsegpt_0.5/no_finetuning \
#     --config_name meta-llama/Llama-3.1-8B \
#     --dataset_name tatsu-lab/alpaca \
#     --num_train_epochs 4 \
#     --block_size 2048 \
#     --torch_dtype bfloat16 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --do_train \
#     --do_eval \
#     --loss_type CrossEntropy \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/howon010402/HighSparsityPruning/llm_weights/llama-3.1-8B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_CrossEntropy
    
# python finetune_dense_lm.py \
#     --model_name_or_path /home/howon010402/HighSparsityPruning/llm_weights/llama-3.1-8B/sparsegpt_0.5/no_finetuning \
#     --config_name meta-llama/Llama-3.1-8B \
#     --dataset_name tatsu-lab/alpaca \
#     --num_train_epochs 4 \
#     --block_size 2048 \
#     --torch_dtype bfloat16 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --do_train \
#     --do_eval \
#     --loss_type SquareHead \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/howon010402/HighSparsityPruning/llm_weights/llama-3.1-8B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_SquareHead
    
# python finetune_dense_lm.py \
#     --model_name_or_path ../llm_weights/llama-3.1-8B/sparsegpt_0.5/no_finetuning \
#     --config_name meta-llama/Llama-3.1-8B \
#     --dataset_name tatsu-lab/alpaca \
#     --num_train_epochs 4 \
#     --block_size 2048 \
#     --torch_dtype bfloat16 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --do_train \
#     --do_eval \
#     --loss_type KLDiv \
#     --learning_rate 1e-4 \
#     --use_paser \
#     --num_clusters 10 \
#     --max_selected_data 20000 \
#     --overwrite_output_dir \
#     --output_dir ../llm_weights/llama-3.1-8B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_KLDiv
python finetune_dense_lm.py \
    --model_name_or_path ../llm_weights/llama-3.1-8B-Instruct/wanda_0.5/no_finetuning \
    --config_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/tulu-3-sft-mixture \
    --num_train_epochs 1 \
    --block_size 2048 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --loss_type CrossEntropy \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir ../llm_weights/llama-3.1-8B-Instruct/wanda_0.5/dense_ft_bf16_1e-4_tulu3_CrossEntropy
    
python finetune_dense_lm.py \
    --model_name_or_path ../llm_weights/llama-3.1-8B-Instruct/wanda_0.5/no_finetuning \
    --config_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/tulu-3-sft-mixture \
    --num_train_epochs 1 \
    --block_size 2048 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --loss_type KLDiv \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir ../llm_weights/llama-3.1-8B-Instruct/wanda_0.5/dense_ft_bf16_1e-4_tulu3_KLDiv
  
# python finetune_dense_lm.py \
#     --model_name_or_path ../llm_weights/llama-3.1-8B/wanda_0.5/no_finetuning \
#     --config_name meta-llama/Llama-3.1-8B \
#     --dataset_name allenai/tulu-3-sft-mixture \
#     --num_train_epochs 1 \
#     --block_size 2048 \
#     --torch_dtype bfloat16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --do_train \
#     --do_eval \
#     --loss_type SquareHead \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir ../llm_weights/llama-3.1-8B/wanda_0.5/dense_ft_bf16_1e-4_tulu3_SquareHead

