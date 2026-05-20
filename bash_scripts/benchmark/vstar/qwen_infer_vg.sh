#!/bin/bash
#SBATCH --job-name=infer_vstar_qwen_vg
#SBATCH --output=./slurm_logs/infer_vstar_qwen_vg_%A.out
#SBATCH --error=./slurm_logs/infer_vstar_qwen_vg_%A.err
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

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

CONFIG_PATH=""
python scripts/benchmark/vstar/qwen_infer_vg.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
