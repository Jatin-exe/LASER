#!/bin/bash
set -e

echo "Choose a model size to test:"
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

# Determine best model file to use
MODEL_DIR="output/dfine_hgnetv2_${model}_custom"
MODEL_STAGE_1="$MODEL_DIR/best_stg1.pth"
MODEL_STAGE_2="$MODEL_DIR/best_stg2.pth"

# Check if either model file exists
if [[ -f "$MODEL_STAGE_2" ]]; then
    BEST_MODEL="$MODEL_STAGE_2"
elif [[ -f "$MODEL_STAGE_1" ]]; then
    BEST_MODEL="$MODEL_STAGE_1"
else
    echo "No trained model found in $MODEL_DIR."
    echo "Please download the pretrained weights or train the model first."
    exit 1
fi

echo "Testing model checkpoint: $BEST_MODEL"

python test_trained_model.py \
    --config_path configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
    --model_path "$BEST_MODEL"

chmod -R 777 /dataset
