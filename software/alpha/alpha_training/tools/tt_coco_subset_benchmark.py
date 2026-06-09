#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
D_FINE_ROOT = PROJECT_ROOT / "D-FINE"
for path in (PROJECT_ROOT, D_FINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.core import YAMLConfig
from src.data.dataset import mscoco_label2category
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


CHECKPOINTS = {
    "s_coco": (
        "D-FINE-S COCO",
        "configs/dfine/dfine_hgnetv2_s_coco.yml",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
    ),
    "s_obj2coco": (
        "D-FINE-S Objects365+COCO",
        "configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth",
    ),
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}")
    urlretrieve(url, path)


def ensure_coco_subset(root: Path, max_images: int) -> tuple[Path, Path]:
    ann_dir = root / "annotations"
    ann_file = ann_dir / "instances_val2017.json"
    if not ann_file.exists():
        zip_path = root / "annotations_trainval2017.zip"
        download(COCO_ANN_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("annotations/instances_val2017.json", root)

    coco = COCO(str(ann_file))
    image_ids = sorted(coco.getImgIds())[:max_images]
    images_dir = root / "val2017"
    for img in coco.loadImgs(image_ids):
        download(COCO_IMAGE_URL.format(file_name=img["file_name"]), images_dir / img["file_name"])

    keep = set(image_ids)
    full = json.loads(ann_file.read_text())
    subset = {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "categories": full["categories"],
        "images": [img for img in full["images"] if img["id"] in keep],
        "annotations": [ann for ann in full["annotations"] if ann["image_id"] in keep],
    }
    subset_file = ann_dir / f"instances_val2017_first{max_images}.json"
    subset_file.write_text(json.dumps(subset))
    return images_dir, subset_file


def load_ttnn_model(config_path: Path, checkpoint_path: Path, device_id: int) -> DFINE_TTNN:
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    return DFINE_TTNN(cfg.model.eval(), device_id=device_id)


def evaluate_model(
    model: DFINE_TTNN,
    images_dir: Path,
    annotations: Path,
    max_images: int,
    score_threshold: float,
    warmup: int,
    output_json: Path,
) -> dict:
    coco = COCO(str(annotations))
    image_infos = coco.loadImgs(sorted(coco.getImgIds())[:max_images])
    predictions = []
    times = []

    with torch.no_grad():
        for idx, img in enumerate(image_infos):
            image_path = images_dir / img["file_name"]
            if idx < warmup:
                _ = model(str(image_path), score_threshold=score_threshold)

            start = time.perf_counter()
            out = model(str(image_path), score_threshold=score_threshold)
            times.append(time.perf_counter() - start)

            labels = out["labels"].to(torch.int64).tolist()
            boxes = out["boxes"].to(torch.float32).tolist()
            scores = out["scores"].to(torch.float32).tolist()
            for label, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box
                predictions.append(
                    {
                        "image_id": img["id"],
                        "category_id": int(mscoco_label2category[int(label)]),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions))

    if predictions:
        coco_dt = coco.loadRes(str(output_json))
        evaluator = COCOeval(coco, coco_dt, "bbox")
        evaluator.params.imgIds = [img["id"] for img in image_infos]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats.tolist()
    else:
        stats = [0.0] * 12

    return {
        "images": len(image_infos),
        "detections": len(predictions),
        "score_threshold": score_threshold,
        "latency_ms_mean": 1000.0 * sum(times) / max(len(times), 1),
        "latency_ms_min": 1000.0 * min(times) if times else 0.0,
        "latency_ms_max": 1000.0 * max(times) if times else 0.0,
        "coco_ap": stats[0],
        "coco_ap50": stats[1],
        "coco_ap75": stats[2],
        "coco_ar100": stats[8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="s_coco")
    parser.add_argument("--data_root", default="data/coco2017_subset")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="output/tt_coco_subset")
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    model_name, rel_config, ckpt_url = CHECKPOINTS[args.model]
    images_dir, ann_file = ensure_coco_subset(Path(args.data_root), args.max_images)

    ckpt_path = Path(args.checkpoint_dir) / Path(ckpt_url).name
    download(ckpt_url, ckpt_path)

    model = load_ttnn_model(D_FINE_ROOT / rel_config, ckpt_path, args.device_id)
    try:
        pred_json = Path(args.output_dir) / f"{args.model}_predictions.json"
        metrics = evaluate_model(
            model,
            images_dir,
            ann_file,
            args.max_images,
            args.score_threshold,
            args.warmup,
            pred_json,
        )
    finally:
        model.close()

    result = {
        "model": model_name,
        "config": str(D_FINE_ROOT / rel_config),
        "checkpoint": str(ckpt_path),
        "device_id": args.device_id,
        "annotations": str(ann_file),
        **metrics,
    }
    result_path = Path(args.output_dir) / f"{args.model}_metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
