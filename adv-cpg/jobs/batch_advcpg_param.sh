#!/bin/bash
#SBATCH --job-name=advcpg
#SBATCH --partition=gpu-v100
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/moprescu/event-privacy/logs/batch_%j.out
#SBATCH --error=/scratch/moprescu/event-privacy/logs/batch_%j.err

module load 2024r1 miniconda3 cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/moprescu/event-privacy/envs/advcpg

cd ~/event-privacy/Adv-CPG

python ~/event-privacy/scripts/run_advcpg_batch.py \
    --input "$INPUT_DIR" \
    --out "$OUT_DIR" \
    --target "$TARGET" \
    --prompt "$PROMPT" \
    --suffix "$SUFFIX"
