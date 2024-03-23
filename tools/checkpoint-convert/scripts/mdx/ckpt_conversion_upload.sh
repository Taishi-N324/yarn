#!/bin/bash

set -e

START=500
END=5000
STEP=500
CHECK_POINT_DIR_PATH=/model/taishi/checkpoint/long-context/Swallow-MS-7b-v0.1/lr_2e-5-minlr_2e-5_warmup_250_seq_16384
BASE_MODEL_CHECKPOINT=tokyotech-llm/Swallow-MS-7b-v0.1
CHECK_CONVERT_POINT_DIR_PATH=/model/taishi/converted-hf-checkpoint/long-context/Swallow-MS-7b-v0.1/lr_2e-5-minlr_2e-5_warmup_250_seq_16384
SL=16384

bash tools/checkpoint-convert/scripts/mdx/convert_deepspeed.sh \
    $START \
    $END \
    $STEP \
    $CHECK_POINT_DIR_PATH

bash tools/checkpoint-convert/scripts/mdx/convert_ckpt.sh \
    $START \
    $END \
    $STEP \
    $CHECK_POINT_DIR_PATH \
    $BASE_MODEL_CHECKPOINT \
    $CHECK_CONVERT_POINT_DIR_PATH \
    $SL

bash tools/checkpoint-convert/scripts/mdx/upload/upload-all.sh \
    $START \
    $END \
    $STEP \
    $CHECK_CONVERT_POINT_DIR_PATH