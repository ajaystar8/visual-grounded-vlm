#!/bin/bash

source /home/${USER}/.bashrc
source activate perception_models

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# disable NCCL P2P: GPU-to-GPU PCIe P2P is broken on some nodes (e.g. d4077), causing TP NCCL init to hang
# export NCCL_P2P_DISABLE=1

MODEL=Qwen/Qwen3-4B-Instruct-2507

# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=1
echo Found $NUM_GPUS GPUs.

vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
