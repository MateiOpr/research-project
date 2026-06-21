#!/bin/bash
#SBATCH --job-name=advcpg_test
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/moprescu/event-privacy/logs/test_%j.out
#SBATCH --error=/scratch/moprescu/event-privacy/logs/test_%j.err

module load 2024r1
module load miniconda3
module load cuda/12.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/moprescu/event-privacy/envs/advcpg

cd ~/event-privacy/Adv-CPG

# default example 
python infer_adv_cpg.py \
  --base_model /scratch/moprescu/event-privacy/models/stable-diffusion-xl-base-1.0 \
  --advcpg_ckpt /scratch/moprescu/event-privacy/models/consistentid/ConsistentID_SDXL-v1.bin \
  --image_encoder /scratch/moprescu/event-privacy/models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --ori_path /home/moprescu/event-privacy/data/-yPwW5V4mhI_0.png \
  --target_path /home/moprescu/event-privacy/Adv-CPG/examples/scarlett_johansson.jpg \
  --out /scratch/moprescu/event-privacy/outputs/test_girl_scarlett.png
