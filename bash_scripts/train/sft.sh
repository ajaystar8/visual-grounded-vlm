#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --output=./slurm_logs/sft_train_%A.out
#SBATCH --error=./slurm_logs/sft_train_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:3

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

export OMP_NUM_THREADS=1

CONFIG_FILE=config/train/sft_qwen25vl7b.yaml

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
torchrun --nproc_per_node="$NPROC" scripts/train/sft.py --config "$CONFIG_FILE"

echo "End time: $(date)"
echo "Done!"
