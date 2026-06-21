#!/bin/bash
#SBATCH --job-name=test_fix
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/moprescu/event-privacy/logs/test_fix_%j.out
#SBATCH --error=/scratch/moprescu/event-privacy/logs/test_fix_%j.err

set -uo pipefail
module load 2024r1 miniconda3 cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/moprescu/event-privacy/envs/advcpg

echo "===== diagnostic ====="
python - << 'PY'
import torch
print('torch       :', torch.__version__)
print('built w/cuda:', torch.version.cuda)
print('arch list   :', torch.cuda.get_arch_list())
print('device      :', torch.cuda.get_device_name(0))
print('matmul ok   :', (torch.randn(8, device='cuda') @ torch.randn(8, device='cuda')).item())
PY

echo
echo "===== one-image Adv-CPG test ====="
cd ~/event-privacy/Adv-CPG
TEST_IMG=$(ls /scratch/moprescu/event-privacy/data/reconstructed_rgb_10/*.png | head -n 1)
echo "Test image: $TEST_IMG"
CUDA_LAUNCH_BLOCKING=1 python infer_adv_cpg.py \
    --base_model /scratch/moprescu/event-privacy/models/stable-diffusion-xl-base-1.0 \
    --advcpg_ckpt /scratch/moprescu/event-privacy/models/consistentid/ConsistentID_SDXL-v1.bin \
    --image_encoder /scratch/moprescu/event-privacy/models/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --ori_path "$TEST_IMG" \
    --target_path /scratch/moprescu/event-privacy/data/albert_einstein.png \
    --prompt "A close-up portrait of a person, natural lighting" \
    --out /scratch/moprescu/event-privacy/outputs/fix_test.png

echo "Exit: $?"
ls -la /scratch/moprescu/event-privacy/outputs/fix_test.png 2>/dev/null || echo "No output."
