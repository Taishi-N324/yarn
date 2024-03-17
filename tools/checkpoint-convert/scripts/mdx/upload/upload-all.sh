#!/bin/bash

set -e

cd /model/taishi/yarn
source venv/bin/activate
start=500
end=5000
increment=500

upload_base_dir=/model/taishi/converted-hf-checkpoint/long-context/Swallow-7b-hf/lr_2e-5-minlr_2e-5_warmup_250_seq_16384/



for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/llama/configuration_llama.py"  "$upload_dir/"
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/llama/modeling_llama_together_yarn.py"  "$upload_dir/"
  cp -r "/model/taishi/tokenizer/Swallow/tokenizer.model" "$upload_dir/tokenizer.model"
  cp -r "/model/taishi/yarn/tools/checkpoint-convert/scripts/configs/Swallow/Swallow-7b-hf-16k-config.json" "$upload_dir/config.json"

  python tools/checkpoint-convert/scripts/mdx/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-7b-hf-yarn-16k-$(printf "%07d" $i)
done
