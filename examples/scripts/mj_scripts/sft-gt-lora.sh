set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset talzoomanzoo/aime_train \
   --input_key Question \
   --output_key answer \
   --train_batch_size 8 \
   --micro_train_batch_size 2 \
   --max_samples 933 \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   --save_path ./checkpoint/deepseek-7b-sft-gt-lora \
   --save_steps 500 \
   --logging_steps 10 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 12 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --lora_rank 64 \
   --lora_alpha 64 \
   --packing_samples \
   --aux_loss_coef 0.0 \
   --ds_tensor_parallel_size 4 \
   --use_wandb \
   --wandb_org "mjgwak" \
   --wandb_project "aime-sft-gt-lora" \
   --wandb_run_name "deepseek-7b-sft-gt-lora-run-$(date +%Y%m%d%H%M%S)"
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
