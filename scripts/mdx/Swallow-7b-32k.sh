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
SCALING_FACTOR=8
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
TOKENIZER_MODEL=/model/taishi/tokenizer/Swallow/tokenizer.model
CHECKPOINT_SAVE_DIR="/model/taishi/checkpoint/long-context/Swallow-7b-hf/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

DATA_PATH="${DATA_PATH} 10605477142 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/split_0_text_document"
DATA_PATH="${DATA_PATH} 10464907226 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/split_1_text_document"
DATA_PATH="${DATA_PATH} 12465407213 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/split_2_text_document"
DATA_PATH="${DATA_PATH} 16446568076 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/split_3_text_document"
DATA_PATH="${DATA_PATH} 38345096470 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/split_4_text_document"
DATA_PATH="${DATA_PATH} 1672543873 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/ja_wiki_merged_train_text_document"
DATA_PATH="${DATA_PATH} 5000000000 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/lumi_en_arxiv_merged_text_document"
DATA_PATH="${DATA_PATH} 5000000000 /model/taishi/datasets/okazaki_lab_cc_1500_okazaki_lab_cc_nfkc_16k_aligned_8/lumi_en_falcon_merged_threadripper-3960x_8_text_document"

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
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
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
  --wandb-entity "prj-jalm" \
  --wandb-project "long-context-experiment" \
  --wandb-name "${JOB_NAME}" \
  --base-model tokyotech-llm/Swallow-7b-hf \
  --scaling-factor $SCALING_FACTOR \
  --architecture llama \

