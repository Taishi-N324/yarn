#!/bin/bash

set -e

source venv/bin/activate

START=$1
END=$2
STEP=$3
CHECK_CONVERT_POINT_DIR_PATH=$4




for ((i = $START; i <= $END; i += $STEP)); do
  upload_dir=$CHECK_CONVERT_POINT_DIR_PATH/iter_$(printf "%07d" $i)
  mkdir -p $upload_dir
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/mistral/configuration_mistral.py"  "$upload_dir/"
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/mistral/modeling_mistral_yarn.py"  "$upload_dir/"
  cp -r "/model/taishi/tokenizer/Swallow-MS-7b-v0.1/tokenizer.model" "$upload_dir/tokenizer.model"
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/Swallow/Swallow-MS-7b-v0.1-16k-config.json" "$upload_dir/config.json"

  python tools/checkpoint-convert/scripts/mdx/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-MS-7b-v0.1-yarn-16k-iter$(printf "%07d" $i)
done