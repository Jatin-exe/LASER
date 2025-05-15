#!/bin/bash
set -e

echo "Choose a model size to export:"
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

# Determine best model file to use
MODEL_DIR="output/dfine_hgnetv2_${model}_custom"
MODEL_STAGE_1="$MODEL_DIR/best_stg1.pth"
MODEL_STAGE_2="$MODEL_DIR/best_stg2.pth"

# Check for model availability
if [[ -f "$MODEL_STAGE_2" ]]; then
    BEST_MODEL="$MODEL_STAGE_2"
elif [[ -f "$MODEL_STAGE_1" ]]; then
    BEST_MODEL="$MODEL_STAGE_1"
else
    echo "No pretrained model found in $MODEL_DIR."
    echo "Please download the pretrained weights or train from scratch."
    exit 1
fi

echo "Exporting ONNX from: $BEST_MODEL"
python tools/deployment/export_onnx.py \
    --check \
    -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
    -r "$BEST_MODEL"

chmod -R 777 /workspace/output
