#!/bin/bash
#SBATCH --job-name=propagate_sam3_masks_vstar
#SBATCH --output=./slurm_logs/propagate_sam3_masks_vstar_%A.out
#SBATCH --error=./slurm_logs/propagate_sam3_masks_vstar_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

OUTPUT_DIR="/scratch/rajendrakumar.a/visual-grounded-vlm/annotations/vstar"
EXPERIMENT_NAME="uniform_sample_cap_768"

python src/dataset/propagate_sam3_masks.py \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME"

echo "End time: $(date)"
echo "Done!"
