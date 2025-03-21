#!/bin/bash

# python main.py \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --prune_method wanda \
#     --sparsity_ratio 0.7 \
#     --sparsity_type unstructured \
#     --save out/Llama-3.1-8B/unstructured/
# python main.py \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --prune_method wanda \
#     --sparsity_ratio 0.8 \
#     --sparsity_type unstructured \
#     --save out/Llama-3.1-8B/unstructured/
# python main.py \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --prune_method sparsegpt \
#     --sparsity_ratio 0.5 \
#     --sparsity_type unstructured \
#     --save out/Llama-3.1-8B/unstructured/
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --prune_method sparsegpt \
    --sparsity_ratio 0.6 \
    --sparsity_type unstructured \
    --save lm_eval/llama-3.1-8B/sparsegpt_0.6/iterative10_no_finetuning/ \
    --save_model llm_weights/llama3.1-8B/sparsegpt_0.6/iterative10_no_finetuning/
python main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save lm_eval/llama-3.1-8B-Instruct/sparsegpt_0.5/no_finetuning/ \
    --save_model llm_weights/llama-3.1-8B-Instruct/sparsegpt_0.5/no_finetuning/
# python main.py \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --prune_method sparsegpt \
#     --sparsity_ratio 0.7 \
#     --sparsity_type unstructured \
#     --save out/Llama-3.1-8B/unstructured/
# python main.py \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --prune_method sparsegpt \
#     --sparsity_ratio 0.8 \
#     --sparsity_type unstructured \
#     --save lm_eval/Llama-3.1-8B/unstructured/