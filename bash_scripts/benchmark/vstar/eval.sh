#!/bin/bash
#SBATCH --job-name=vstar_benchmark
#SBATCH --output=./slurm_logs/new_vstar_eval_%A.out
#SBATCH --error=./slurm_logs/new_vstar_eval_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:2

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

EXP_DIR=/scratch/rajendrakumar.a/visual-grounded-vlm/results/qwen25vl7bi_vstar_zeroshot/2026-05-18_18-35-15
python scripts/benchmark/vstar/eval.py --exp_dir "$EXP_DIR"

echo "End time: $(date)"
echo "Done!"
