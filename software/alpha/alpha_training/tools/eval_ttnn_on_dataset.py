#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


def load_coco_annotations(path: Path):
    data = json.loads(Path(path).read_text())
    images = {img["id"]: img["file_name"] for img in data["images"]}
    gt_by_filename = {}
    for ann in data["annotations"]:
        fname = images.get(ann["image_id"])  # xywh format
        if fname is None:
            continue
        gt_by_filename.setdefault(fname, []).append(ann["bbox"])  # [x,y,w,h]
    return gt_by_filename


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [M,4], b:[N,4]
    M, N = a.shape[0], b.shape[0]
    if M == 0 or N == 0:
        return torch.zeros((M, N))
    a = a.unsqueeze(1).expand(M, N, 4)
    b = b.unsqueeze(0).expand(M, N, 4)
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def main():
    p = argparse.ArgumentParser("Evaluate DFINE TTNN on dataset with IoU@0.5 precision/recall")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--max_images", type=int, default=50)
    p.add_argument("--score_threshold", type=float, default=0.4)
    p.add_argument("--output", type=str, default="output/ttnn_eval.txt")
    args = p.parse_args()

    cfg = YAMLConfig(args.config_path, resume=args.model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model_pt = cfg.model.eval()
    model_tt = DFINE_TTNN(model_pt, device_id=args.device_id)

    gt = load_coco_annotations(Path(args.annotations))
    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    img_paths = img_paths[: args.max_images]

    total_tp = total_fp = total_fn = 0
    per_image_stats = []
    for img_path in img_paths:
        file_name = img_path.name
        out = model_tt(str(img_path), score_threshold=args.score_threshold)
        boxes_xyxy = out["boxes"]  # [K,4]
        scores = out["scores"]
        # GT xywh -> xyxy
        gt_xywh = torch.tensor(gt.get(file_name, []), dtype=torch.float32)
        if gt_xywh.numel() == 0:
            gt_xyxy = torch.zeros((0, 4))
        else:
            x1y1 = gt_xywh[:, :2]
            x2y2 = gt_xywh[:, :2] + gt_xywh[:, 2:]
            gt_xyxy = torch.cat([x1y1, x2y2], dim=1)

        # Greedy matching at IoU>=0.5
        ious = iou_xyxy(boxes_xyxy, gt_xyxy)
        matched_gt = set()
        tp = 0
        for pi in torch.argsort(scores, descending=True):
            if boxes_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
                break
            giou = ious[pi]
            if giou.numel() == 0:
                continue
            gj = int(torch.argmax(giou).item())
            if giou[gj].item() >= 0.5 and gj not in matched_gt:
                tp += 1
                matched_gt.add(gj)
        fp = int(max(0, boxes_xyxy.shape[0] - tp))
        fn = int(max(0, gt_xyxy.shape[0] - tp))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image_stats.append((file_name, tp, fp, fn))

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("DFINE_TTNN evaluation (IoU>=0.5)\n")
        f.write(f"Images: {len(img_paths)} ScoreThr: {args.score_threshold}\n")
        f.write(f"TP={total_tp} FP={total_fp} FN={total_fn}\n")
        f.write(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}\n")
    print(f"Wrote evaluation to {out_path}")

    try:
        model_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

