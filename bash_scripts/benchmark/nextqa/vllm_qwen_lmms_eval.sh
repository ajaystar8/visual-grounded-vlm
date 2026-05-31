#!/bin/bash
#SBATCH --job-name=vllm_lmmseval_nextqa_qwen
#SBATCH --output=./slurm_logs/vllm_lmmseval_nextqa_qwen_%A.out
#SBATCH --error=./slurm_logs/vllm_lmmseval_nextqa_qwen_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:h200:1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

source /home/${USER}/.bashrc
source activate perception_models
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# lmms-eval lives in a separate repo from visual-grounded-vlm
LMMS_EVAL_ROOT=/projects/torresani-lab/ajay/lmms-eval
cd $LMMS_EVAL_ROOT

MODEL=Qwen/Qwen2.5-VL-7B-Instruct
TASK=nextqa_mc_test
BATCH_SIZE=32

# Timestamped output dir so runs don't overwrite each other
RUN_NAME="qwen25vl7bi_${TASK}_zeroshot"
OUTPUT_PATH=/scratch/rajendrakumar.a/visual-grounded-vlm/results/lmmseval/${RUN_NAME}/$(date +%Y%m%d_%H%M%S)

echo "Output path: $OUTPUT_PATH"

python3 -m lmms_eval \
    --model vllm \
    --model_args model=$MODEL,tensor_parallel_size=1 \
    --tasks $TASK \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path $OUTPUT_PATH \
    --verbosity INFO

echo "End time: $(date)"
echo "Done!"
