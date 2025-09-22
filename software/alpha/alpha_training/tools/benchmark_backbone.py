#!/usr/bin/env python3
import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Make project root importable regardless of CWD
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig


def load_images(images_dir: Path, size: int) -> List[torch.Tensor]:
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    imgs: List[torch.Tensor] = []
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sorted(images_dir.glob(ext)))
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = tfm(img).unsqueeze(0)
        imgs.append(t)
    return imgs


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def build_pt_backbone(cfg_path: str, model_path: str):
    cfg = YAMLConfig(cfg_path, resume=model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    backbone = cfg.model.backbone.eval()
    return backbone


def run_pt(backbone, images: List[torch.Tensor], num_images: int, warmup_runs: int) -> List[float]:
    latencies: List[float] = []
    with torch.no_grad():
        # Warmup on the first image
        x0 = images[0]
        for _ in range(max(0, warmup_runs)):
            _ = backbone(x0)

        for i in range(num_images):
            x = images[i % len(images)]
            t0 = time.perf_counter()
            _ = backbone(x)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
    return latencies


def run_ttnn(backbone_pt, images: List[torch.Tensor], num_images: int, warmup_runs: int) -> List[float]:
    from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual

    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=0)
    latencies: List[float] = []
    ttnn = backbone_tt.ttnn

    try:
        # Warmup on the first image
        x0 = images[0]
        for _ in range(max(0, warmup_runs)):
            _ = backbone_tt(x0)
            try:
                ttnn.synchronize_device(backbone_tt.device)
            except Exception:
                pass

        for i in range(num_images):
            x = images[i % len(images)]
            t0 = time.perf_counter()
            _ = backbone_tt(x)
            # Ensure device has finished work before stopping timer
            try:
                ttnn.synchronize_device(backbone_tt.device)
            except Exception:
                pass
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
    finally:
        try:
            backbone_tt.close()
        except Exception:
            pass
    return latencies


def summarize_and_print(backend: str, latencies_s: List[float]) -> None:
    n = len(latencies_s)
    total_time = sum(latencies_s)
    avg_ms = (total_time / n) * 1000.0 if n else 0.0
    med_ms = statistics.median(latencies_s) * 1000.0 if n else 0.0
    p95_ms = percentile(latencies_s, 0.95) * 1000.0 if n else 0.0
    thr = (n / total_time) if total_time > 0 else 0.0

    print("\n=== Backbone Benchmark Summary ===")
    print(f"Backend Tested     : {backend}")
    print(f"Total Timed Runs   : {n}")
    print(f"Average Latency    : {avg_ms:.3f} ms")
    print(f"Median Latency     : {med_ms:.3f} ms")
    print(f"P95 Latency        : {p95_ms:.3f} ms")
    print(f"Throughput         : {thr:.2f} img/s")


def main():
    p = argparse.ArgumentParser("Benchmark HGNetv2 backbone on PyTorch or TTNN")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--backend", required=True, choices=["pytorch", "ttnn"])
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--warmup_runs", type=int, default=10)
    p.add_argument("--size", type=int, default=512, help="Resize inputs to size x size (default: 512)")
    args = p.parse_args()

    images = load_images(Path(args.images_dir), size=args.size)

    import time
    if args.backend == "pytorch":
        backbone_pt = build_pt_backbone(args.config_path, args.model_path)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = backbone_pt(images[0])
        t_init = time.perf_counter() - t0
        latencies = run_pt(backbone_pt, images, args.num_images, args.warmup_runs)
        print(f"Init + First Run (PT): {t_init*1000:.2f} ms")
        summarize_and_print(args.backend, latencies)
    else:
        backbone_pt = build_pt_backbone(args.config_path, args.model_path)
        t0 = time.perf_counter()
        latencies = run_ttnn(backbone_pt, images, args.num_images, args.warmup_runs)
        t_init = 0.0  # encompassed by run_ttnn warmup; detailed per-op compile varies
        print(f"Init + First Run (TTNN): (measured in warmup)")
        summarize_and_print(args.backend, latencies)


if __name__ == "__main__":
    main()
