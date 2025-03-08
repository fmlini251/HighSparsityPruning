#!/bin/sh

# MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_SquareHead/checkpoint-76"
# RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_SquareHead"
# MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_CrossEntropy/checkpoint-76"
# RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_CrossEntropy"
# MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_KLDiv/checkpoint-75"
# RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca_KLDiv"
# MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/sparsegpt_0.5_llama3.2-1B"
# RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/sparsegpt_0.5/no_finetuning"
MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/original"
RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/original"
echo "Start Evaluation"

# run evaluation
CUDA_VISIBLE_DEVICES=3 script -efq ${RESULT_DIR}/perplexity.log -c \
    "accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL_DIR} \
    --tasks wikitext \
    --batch_size auto \
    --output_path ${RESULT_DIR}"

CUDA_VISIBLE_DEVICES=3 script -efq ${RESULT_DIR}/commonsense_zero_shot.log -c \
    "accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL_DIR} \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size auto \
    --num_fewshot 0 \
    --trust_remote_code \
    --output_path ${RESULT_DIR}"