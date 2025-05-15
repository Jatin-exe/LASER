#!/bin/bash

echo "Choose a model size to train:"
echo "[n] nano"
echo "[s] small"
echo "[m] medium"
echo "[l] large"
echo "[x] extra-large"
read -p "Enter model size (n/s/m/l/x): " model

# Validate input
if [[ ! "$model" =~ ^(n|s|m|l|x)$ ]]; then
    echo "❌ Invalid model size: '$model'"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))

# Echo training configuration
echo "Training DFine-HGNetV2 [$model] using $NUM_GPUS GPU(s): $GPU_IDS"

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --master_port=7777 --nproc_per_node=$NUM_GPUS train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0

chmod -R 777 /workspace/output