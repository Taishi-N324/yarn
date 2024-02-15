#!/bin/bash

source venv/bin/activate

accelerate launch --config_file configs/accelerate_config_zero3.yaml finetune.py \
    --output-dir output/linear-7b-8k \
    --model NousResearch/Llama-2-7b-hf \
    --scaling-type linear \
    --scaling-factor 2 \
    --deepspeed \
    --architecture llama \
    --wandb "yarn" \
    --dataset output/truncated-8k \
    --max-train-steps 1000 \