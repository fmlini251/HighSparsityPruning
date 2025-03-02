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
    --model meta-llama/Meta-Llama-3.1-8B \
    --prune_method sparsegpt \
    --sparsity_ratio 0.6 \
    --sparsity_type unstructured \
    --save out/Llama-3.1-8B/unstructured/
python main.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --prune_method sparsegpt \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/Llama-3.1-8B/unstructured/
python main.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --prune_method sparsegpt \
    --sparsity_ratio 0.8 \
    --sparsity_type unstructured \
    --save out/Llama-3.1-8B/unstructured/