#!/bin/bash

# swich virtual env
source venv/bin/activate

JOB_ID=$(date +%s%N)
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12803

echo "MASTER_ADDR=${MASTER_ADDR}"
# hostfile
HOSTFILE_NAME=scripts/mdx/hostfile/hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"

NHOSTS=16
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# training config
SCALING_FACTOR=4
SEQ_LENGTH=$((4096 * SCALING_FACTOR))
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
TRAIN_STEPS=5000  # 5,000 iteration = 10B Token

# optimizer config
LR=2e-5
MIN_LR=2e-5
LR_WARMUP_STEPS=250
LR_DECAY_STEPS=5000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/model/llmjp0/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR="/model/yarn-checkpoint/13b-cc-v2-beta-61000step.code20K_en40K_ja60K_ver2.2/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
# data config
DATASET_DIR=/data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K

TRAIN_DATA_PATH=""
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1442858730 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki/ja_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4697570389 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_wiki/en_wiki_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8670888578 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/code_stack/code_stack_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16802680411 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16814033199 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16941076697 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16933151184 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16863221834 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 16928046076 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8477979936 /data/llm-jp-corpus/v1.0.1/merge/binarized/ver2.2/code20K_en40K_ja60K/en_pile/en_pile_merge_7_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 485412592.3 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 671720325.2 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1520327461 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2210405544 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1950073735 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2447399807 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1607587452 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2614097085 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1936833706 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2072378019 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1085097659 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2017-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1520398698 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2821380916 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1386476874 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1365024447 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1079992869 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3154362631 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3508566438 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2526846091 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2568027376 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2795873847 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2691462512 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3000093885 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2018-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2623504180 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2809876295 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-09.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2200639273 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-13.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2094244433 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-18.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2099166494 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-22.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1974100479 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-26.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1881839207 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-30.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2135269364 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-35.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1798071010 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1996550453 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1719580748 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-47.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1603557847 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2019-51.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2099920626 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2067348539 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1993241361 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-16.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2029791266 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-24.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2322944978 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-29.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1965010132 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-34.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2479626171 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1800491331 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-45.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1675306449 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2020-50.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2155870225 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-04.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1835666333 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-10.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2059578946 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-17.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1805208879 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1562020823 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-25.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1776641448 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-31.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1867977822 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-39.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2291113661 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-43.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2296646892 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2021-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2849405378 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-05.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2730361649 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-21.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2417978889 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-27.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2101837374 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-33.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1955769700 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-40.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2110014918 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2022-49.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2197497215 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2023-06.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1827420392 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2023-14.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1800224491 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2023-23.jsonl_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2016585896 /data/llm-jp-corpus-v2.1-CC/binarized/ver2.2/code20K_en40K_ja60K/CC-ver2.1/CC-MAIN-2013-2016.jsonl_text_document"

# job name
JOB_NAME="Swallow-7b-hf-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"



mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  python finetune.py \
  --seq-length ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-6 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config "deepspeed/zero3_offload.json" \
  --zero-stage 3 \
  --use-mpi \
  --wandb-entity "llm-jp" \
  --wandb-project "yarn-long-context-experiment" \
  --wandb-name "${JOB_NAME}" \
  --base-model /model/13B_HF/llm-jp-13b-cc-v2-beta-61000step.code20K_en40K_ja60K_ver2.2/ \
  --scaling-factor $SCALING_FACTOR \
  --architecture llama \

