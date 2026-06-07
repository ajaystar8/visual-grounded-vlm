#!/bin/bash
#SBATCH --job-name=openo3_str_sam3_track
#SBATCH --output=./slurm_logs/openo3_str_sam3_track_%A.out
#SBATCH --error=./slurm_logs/openo3_str_sam3_track_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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

CONFIG_PATH="config/track/openo3_str_sam3_track.yaml"
python src/dataset/openo3/sam3_track.py --config $CONFIG_PATH

echo "End time: $(date)"
echo "Done!"
