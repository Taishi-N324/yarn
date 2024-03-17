#!/bin/bash

set -e

# swich virtual env
source venv/bin/activate

start=500
end=5000
increment=500
BASE_MODEL_CHECKPOINT=tokyotech-llm/Swallow-7b-hf

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/model/taishi/checkpoint/long-context/Swallow-7b-hf/lr_2e-5-minlr_2e-5_warmup_250_seq_16384/${FORMATTED_ITERATION}/fp32_model.bin
  OUTPUT_PATH=/model/taishi/converted-hf-checkpoint/long-context/Swallow-7b-hf/lr_2e-5-minlr_2e-5_warmup_250_seq_16384/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 16384
done
