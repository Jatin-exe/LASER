#!/bin/bash
set -e

echo "Choose a model size to train:"
echo "[n] nano"
echo "[s] small"
echo "[m] medium"
echo "[l] large"
echo "[x] extra-large"
read -p "Enter model size (n/s/m/l/x): " model

# Validate input
if [[ ! "$model" =~ ^(n|s|m|l|x)$ ]]; then
    echo "Invalid model size: '$model'"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))

MODEL_DIR="output/dfine_hgnetv2_${model}_custom"
MODEL_STAGE_1="$MODEL_DIR/best_stg1.pth"
MODEL_STAGE_2="$MODEL_DIR/best_stg2.pth"

# Check for existing model files
if [[ -f "$MODEL_STAGE_2" ]]; then
    BEST_MODEL="$MODEL_STAGE_2"
elif [[ -f "$MODEL_STAGE_1" ]]; then
    BEST_MODEL="$MODEL_STAGE_1"
else
    echo "No pretrained model found in $MODEL_DIR."
    echo "Please download the pretrained weights or train from scratch."
    exit 1
fi

echo "Using model checkpoint: $BEST_MODEL"
echo "Tuning DFine-HGNetV2 [$model] using $NUM_GPUS GPU(s): $GPU_IDS"

# Launch training
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --master_port=7777 \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
    --use-amp \
    --seed=0 \
    -t "$BEST_MODEL"

# Permissions fix (for Docker or shared volume use cases)
chmod -R 777 /workspace/output
