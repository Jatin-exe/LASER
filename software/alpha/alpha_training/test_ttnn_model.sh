#!/usr/bin/env bash
set -euo pipefail

python -m tools.run_ttnn_inference \
  --image_path "${1:-dataset/images/val/1713903332_212967648.jpg}" \
  --model_path "${2:-output/dfine_hgnetv2_n_custom/best_stg1.pth}" \
  --config_path "${3:-dfine_hgnetv2_n_custom.yml}" \
  --output "${4:-output/ttnn_infer.jpg}"

