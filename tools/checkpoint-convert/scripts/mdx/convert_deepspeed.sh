#!/bin/bash
set -e

cd /model/taishi/yarn
# swich virtual env
source venv/bin/activate

START=500
END=5000
STEP=500


for (( ITERATION=$START; ITERATION<=$END; ITERATION+=$STEP ))
do
    FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)
    CHECK_POINT_DIR=/model/taishi/checkpoint/long-context/Swallow-7b-hf/lr_2e-5-minlr_2e-5_warmup_250_seq_16384/${FORMATTED_ITERATION}
    
    python tools/checkpoint-convert/zero_to_fp32.py \
      --checkpoint-dir $CHECK_POINT_DIR \
      --output-file $CHECK_POINT_DIR/fp32_model.bin \
      --debug

    echo "Conversion completed for $FORMATTED_ITERATION"
done