#!/bin/bash

source /home/${USER}/.bashrc
source activate perception_models

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=Qwen/Qwen2.5-VL-72B-Instruct

# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=2
echo Found $NUM_GPUS GPUs.

vllm serve $MODEL \
    --allowed-local-media-path /projects/torresani-lab/ajay/ \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
