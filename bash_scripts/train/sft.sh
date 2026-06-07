#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --output=./slurm_logs/sft_train_%A.out
#SBATCH --error=./slurm_logs/sft_train_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

CONFIG_FILE=config/train/sft_qwen25vl7b_frame_nums.yaml

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
torchrun --nproc_per_node="$NPROC" scripts/train/sft.py --config "$CONFIG_FILE"

echo "End time: $(date)"
echo "Done!"
