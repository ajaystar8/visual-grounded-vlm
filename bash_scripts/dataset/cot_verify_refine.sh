#!/bin/bash
#SBATCH --job-name=cot_verify_refine
#SBATCH --output=./slurm_logs/cot_verify_refine_%A.out
#SBATCH --error=./slurm_logs/cot_verify_refine_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

CONFIG_PATH="config/dataset/cot_verify_refine.yaml"
python src/dataset/openo3/cot_verify_refine.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
