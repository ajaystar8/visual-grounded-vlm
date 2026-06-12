#!/bin/bash
#SBATCH --job-name=openo3_plm_rcap
#SBATCH --output=./slurm_logs/openo3_plm_rcap_%A.out
#SBATCH --error=./slurm_logs/openo3_plm_rcap_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

PROJECT_ROOT=/projects/torresani-lab/ajay/visual-grounded-vlm
cd $PROJECT_ROOT

CONFIG_PATH="config/dataset/openo3_plm_rcap.yaml"
python src/dataset/openo3/plm_rcap.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
