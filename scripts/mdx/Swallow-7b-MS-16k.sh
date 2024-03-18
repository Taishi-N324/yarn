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
SLIDING_WINDOW_ATTENTION_SCHEDULE=$((1 * SEQ_LENGTH))
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
TOKENIZER_MODEL=/model/taishi/tokenizer/Swallow-MS-7b-v0.1/tokenizer.model
CHECKPOINT_SAVE_DIR="/model/taishi/checkpoint/long-context/Swallow-MS-7b-v0.1/lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

DATA_PATH="${DATA_PATH} 9443206541 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"
DATA_PATH="${DATA_PATH} 9319866579 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_1_text_document"
DATA_PATH="${DATA_PATH} 11101717201 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_2_text_document"
DATA_PATH="${DATA_PATH} 14645659270 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_3_text_document"
DATA_PATH="${DATA_PATH} 34160880690 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_4_text_document"
DATA_PATH="${DATA_PATH} 1487598324 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/ja_wiki_merged_text_document"
DATA_PATH="${DATA_PATH} 4453273811 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/arxiv_text_document"
DATA_PATH="${DATA_PATH} 4453273811 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/falcon_text_document"
DATA_PATH="${DATA_PATH} 10934523772 /model/taishi/datasets/mistral_16k_Llama2Tokenizer/algebraic_stack_text_document"

# job name
JOB_NAME="Swallow-MS-7b-v0.1-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"



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
  --base-model tokyotech-llm/Swallow-MS-7b-v0.1 \
  --scaling-factor $SCALING_FACTOR \
  --architecture mistral \
  --original-max-position-embeddings 4096 \
  --sliding-window-attention-schedule $SLIDING_WINDOW_ATTENTION_SCHEDULE \

