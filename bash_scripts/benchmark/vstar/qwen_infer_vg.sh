#!/bin/bash
#SBATCH --job-name=infer_vstar_qwen_vg
#SBATCH --output=./slurm_logs/infer_vstar_qwen_vg_%A.out
#SBATCH --error=./slurm_logs/infer_vstar_qwen_vg_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=d4077

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn # vLLM crashes if not set

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

CONFIG_PATH="config/infer/qwen25vl7b_vstar_sft_vg/bbox.yaml"
python scripts/benchmark/vstar/qwen_infer_vg.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
