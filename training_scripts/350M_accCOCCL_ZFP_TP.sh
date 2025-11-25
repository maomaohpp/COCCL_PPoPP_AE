#!/bin/bash
module load miniconda/24.9.2 cuda/12.4
source activate python3.10
#conda activate SDP4bit
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


export WANDB_API_KEY=89aab7ac312376a880efc7b8081dc55c52dfdb28
export WANDB_MODE=offline
NNODES=$1
NPROC_PER_NODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT="29501"
BATCH_JOB_ID=$5



export NCCL_HOME=/data/home/scyb226/lxc/coccl-acc/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_HOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$NCCL_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_HOME/include:$CPLUS_INCLUDE_PATH

export PATH=/data/home/scyb226/lxc/coccl-acc/src/device/compress/zfp/build/bin:$PATH
export LIBRARY_PATH=/data/home/scyb226/lxc/coccl-acc/src/device/compress/zfp/build/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/data/home/scyb226/lxc/coccl-acc/src/device/compress/zfp/build/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/data/home/scyb226/lxc/coccl-training/logs/350M/ncclcomp.%h
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13

export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit,tahquant,zfp_compressor
export NCCL_ENABLE_ALLTOALL_COMPRESS=0
export NCCL_ALLTOALL_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=zfp_compressor
export NCCL_ENABLE_ALLGATHER_COMPRESS=0
export NCCL_ALLGATHER_COMPRESSORS=zfp_compressor
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=0
export NCCL_REDUCESCATTER_COMPRESSORS=zfp_compressor
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=zfp_compressor
export NCCL_ENABLE_SENDRECV_COMPRESS=0
export NCCL_SENDRECV_COMPRESSORS=zfp_compressor
export NCCL_SENDRECV_BWD_COMPRESSORS=zfp_compressor
export NCCL_COMPRESSORS_CONFIG_PATH=/data/home/scyb226/lxc/coccl-acc/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=/data/home/scyb226/lxc/coccl-acc/build/obj/device/compress/libcompress
# export NCCL_ALGO=Tree
export NCCL_LOCAL_REGISTER=1
export NCCL_CHECKS_DISABLE=1
export NCCL_BUFFSIZE=33554432
# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
OUTPUT_LOG="350M_train_rank${NODE_RANK}_${BATCH_JOB_ID}_tp_zfp_8bit.log"

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=4
MICRO_BATCH_SIZE=8
# Calculate GLOBAL_BATCH_SIZE based on Accumulation Step=1
GLOBAL_BATCH_SIZE=256
CHECKPOINT_PATH=/data/home/scyb226/lxc/coccl-training/checkpoint/350M_Baseline
VOCAB_FILE=/data/home/scyb226/khr/the-pile/vocab.json
MERGE_FILE=/data/home/scyb226/khr/the-pile/merges.txt
DATA_PATH=/data/home/scyb226/khr/the-pile/pile_text_document

TENSORBOARD_DIR=/data/home/scyb226/lxc/coccl-training/tensorboard
WANDB_DIR=/data/home/scyb226/lxc/coccl-training/wandb

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

TRAINING_ARGS="
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 20000 \
"

OPTIMIZER_ARGS="
    --lr 0.0003 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00003 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --bf16 \
    --use-distributed-optimizer \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --timing-log-level 2 \
    --save-interval 10002 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --log-memory-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --wandb-project COCCL-PPoPP \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name 350M-GPT_2-accTPZFP-8bit-32H800 \
"
QUANTIZE_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
"

cd /data/home/scyb226/lxc/SDP4Bit-COCCL

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    --distributed-backend nccl > "/data/home/scyb226/lxc/coccl-training/logs/350M/${OUTPUT_LOG}"
