CUDA_VISIBLE_DEVICES=3 python finetune_lm.py \
    --model_name_or_path /home/howonlee/HighSparsityPruning/llm_weights/sparsegpt_0.5_llama3.2-1B \
    --config_name "meta-llama/Llama-3.2-1B" \
    --dataset_name tatsu-lab/alpaca \
    --num_train_epochs 4 \
    --block_size 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir /home/howonlee/HighSparsityPruning/llm_weights/sparsegpt_0.5_llama3.2-1B_lora_ft

CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
    --model [PATH to load sparse pruned LLaMA-7B] \
    --lora_weights [PATH to load the LoRA weights]