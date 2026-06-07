#!/bin/bash

source /home/${USER}/.bashrc
source activate perception_models

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=Qwen/Qwen2.5-VL-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen3-VL-8B-Thinking

LORA_MODULE="/scratch/rajendrakumar.a/visual-grounded-vlm/sft/checkpoints/sft_qwen25vl7bi/only_kf_red_bbox_frame_nums_sys_prompt_v2"        # set to adapter dir to enable LoRA, e.g. /scratch/.../sft_qwen25vl7bi
LORA_NAME="sft"       # name used when calling the model via API

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo Found $NUM_GPUS GPUs.

if [ -n "$LORA_MODULE" ]; then
    # LoRA requires tensor-parallel-size=1
    vllm serve $MODEL \
        --allowed-local-media-path /projects/torresani-lab/ajay/ \
        --tensor-parallel-size 1 \
        --enable-lora \
        --lora-modules ${LORA_NAME}=${LORA_MODULE}
else
    echo "LORA module path not found. Defaulting to original checkpoint..."
    vllm serve $MODEL \
        --allowed-local-media-path /projects/torresani-lab/ajay/ \
        --tensor-parallel-size $NUM_GPUS
fi