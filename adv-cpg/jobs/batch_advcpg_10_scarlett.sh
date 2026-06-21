#!/bin/bash
#SBATCH --job-name=advcpg_10_sc
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/moprescu/event-privacy/logs/advcpg10sc_%j.out
#SBATCH --error=/scratch/moprescu/event-privacy/logs/advcpg10sc_%j.err

module load 2024r1 miniconda3 cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/moprescu/event-privacy/envs/advcpg

cd ~/event-privacy/Adv-CPG
python ~/event-privacy/scripts/run_advcpg_batch.py \
    --input /scratch/moprescu/event-privacy/data/reconstructed_rgb_10 \
    --out /scratch/moprescu/event-privacy/data/advcpg_10_scarlett_facial \
    --target /scratch/moprescu/event-privacy/data/scarlet.png \
    --prompt "A face portrait, neutral expression, clear lighting" \
    --suffix scarlett_facial
