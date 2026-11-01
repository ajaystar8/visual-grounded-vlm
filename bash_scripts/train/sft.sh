#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --output=./slurm_logs/sft_train_%A.out
#SBATCH --error=./slurm_logs/sft_train_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

# sft_cot.py + sft_qwen25vl7b_openo3_video.yaml: dense-bbox CoT SFT
# sft_bbox.py + sft_qwen25vl7b{,_frame_nums}.yaml: keyframe-bbox SFT
SFT_SCRIPT=${SFT_SCRIPT:-scripts/train/sft_cot.py}
CONFIG_FILE=${CONFIG_FILE:-config/train/sft_qwen25vl7b_openo3_video.yaml}

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
torchrun --nproc_per_node="$NPROC" "$SFT_SCRIPT" --config "$CONFIG_FILE"

echo "End time: $(date)"
echo "Done!"
