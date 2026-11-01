#!/bin/bash
#SBATCH --job-name=openo3_str_cot_synth
#SBATCH --output=./slurm_logs/openo3_str_cot_synth_%A.out
#SBATCH --error=./slurm_logs/openo3_str_cot_synth_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

CONFIG_PATH="config/dataset/openo3_str_cot_synth.yaml"
python src/dataset/openo3/str_cot_synth.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
