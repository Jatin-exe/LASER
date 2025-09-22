#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


def draw_and_save(image_path: Path, boxes: torch.Tensor, scores: torch.Tensor, out_path: Path, score_threshold=0.4):
    im = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # Ensure float32 for OpenCV compatibility
    boxes = boxes.to(torch.float32).cpu().numpy()
    scores = scores.to(torch.float32).cpu().numpy()
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        if s < score_threshold:
            continue
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(im, f"{s:.2f}", (int(x1), max(0, int(y1)-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)


def main():
    p = argparse.ArgumentParser("Run end-to-end DFINE TTNN inference")
    p.add_argument("--image_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--score_threshold", type=float, default=0.4)
    p.add_argument("--output", type=str, default="output/ttnn_infer.jpg")
    args = p.parse_args()

    cfg = YAMLConfig(args.config_path, resume=args.model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model_pt = cfg.model.eval()

    model_tt = DFINE_TTNN(model_pt, device_id=args.device_id)

    out = model_tt(args.image_path, score_threshold=args.score_threshold)
    boxes, scores = out["boxes"], out["scores"]
    draw_and_save(Path(args.image_path), boxes, scores, Path(args.output), score_threshold=args.score_threshold)

    print(f"Saved visualization to {args.output}")
    try:
        model_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
