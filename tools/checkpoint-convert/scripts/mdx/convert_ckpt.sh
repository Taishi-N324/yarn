#!/bin/bash

set -e

# swich virtual env
source venv/bin/activate

START=$1
END=$2
STEP=$3
CHECK_POINT_DIR_PATH=$4
BASE_MODEL_CHECKPOINT=$5
CHECK_CONVERT_POINT_DIR_PATH=$6
SL=$7

for ((i = $START; i <= $END; i += $STEP)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=${CHECK_POINT_DIR_PATH}/${FORMATTED_ITERATION}/fp32_model.bin
  OUTPUT_PATH=${CHECK_CONVERT_POINT_DIR_PATH}/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length $SL
done
