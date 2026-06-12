#!/bin/bash

source /home/${USER}/.bashrc
source activate perception_models

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=Qwen/Qwen2.5-VL-7B-Instruct

LORA_MODULE=""   # set to LoRA checkpoint dir to enable LoRA, e.g. /scratch/.../sft_qwen25vl7bi/...
LORA_NAME="sft"  # name used when calling the model via API

# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=1
echo Found $NUM_GPUS GPUs.

if [ -n "$LORA_MODULE" ]; then
    # LoRA requires tensor-parallel-size=1
    vllm serve $MODEL \
        --allowed-local-media-path /projects/torresani-lab/ajay/ \
        --tensor-parallel-size 1 \
        --enable-lora \
        --lora-modules ${LORA_NAME}=${LORA_MODULE}
else
    vllm serve $MODEL \
        --allowed-local-media-path /projects/torresani-lab/ajay/ \
        --tensor-parallel-size $NUM_GPUS
fi
