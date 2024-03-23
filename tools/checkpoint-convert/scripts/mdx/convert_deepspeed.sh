#!/bin/bash
set -e

# swich virtual env
source venv/bin/activate

START=$1
END=$2
STEP=$3
CHECK_POINT_DIR_PATH=$4


for (( ITERATION=$START; ITERATION<=$END; ITERATION+=$STEP ))
do
    FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)
    CHECK_POINT_DIR=${CHECK_POINT_DIR_PATH}/${FORMATTED_ITERATION}
    
    python tools/checkpoint-convert/zero_to_fp32.py \
      --checkpoint-dir $CHECK_POINT_DIR \
      --output-file $CHECK_POINT_DIR/fp32_model.bin \
      --debug

    echo "Conversion completed for $FORMATTED_ITERATION"
done
