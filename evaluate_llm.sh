#!/bin/sh

MODEL_DIR="/home/howonlee/HighSparsityPruning/llm_weights/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca/checkpoint-5"
RESULT_DIR="/home/howonlee/HighSparsityPruning/lm_eval/llama3.2-1B/sparsegpt_0.5/dense_ft_bf16_1e-4_alpaca/checkpoint-5"

echo "Start Evaluation"

# run evaluation
script -efq ${RESULT_DIR}/perplexity.log -c \
    "accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL_DIR} \
    --tasks wikitext \
    --batch_size auto \
    --output_path ${RESULT_DIR}"

script -efq ${RESULT_DIR}/commonsense_zero_shot.log -c \
    "accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL_DIR} \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size auto \
    --num_fewshot 0 \
    --trust_remote_code \
    --output_path ${RESULT_DIR}"