#!/bin/bash
#SBATCH --job-name=vstar_rcap
#SBATCH --output=./slurm_logs/vstar_rcap_%A.out
#SBATCH --error=./slurm_logs/vstar_rcap_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:2

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn # vLLM crashes if not set

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

CONFIG_PATH="config/dataset/vstar_rcap.yaml"
python src/dataset/vstar/rcap.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
