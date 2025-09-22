#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


def to_tensor(img_path: Path, size):
    img = Image.open(img_path).convert("RGB")
    x = T.Compose([T.Resize(size), T.ToTensor()])(img).unsqueeze(0)
    return x


def main():
    p = argparse.ArgumentParser("Benchmark Full DFINE: PyTorch vs TTNN")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--backend", choices=["both", "pytorch", "ttnn"], default="both")
    args = p.parse_args()

    cfg = YAMLConfig(args.config_path, resume=args.model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model_pt = cfg.model.eval()

    # Build TTNN model and measure init+first-run time
    t_init_start = time.perf_counter()
    model_tt = DFINE_TTNN(model_pt, device_id=args.device_id)

    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    if not img_paths:
        print("No images found in", args.images_dir)
        return

    x = to_tensor(img_paths[0], (args.size, args.size))

    t_pt = None
    if args.backend in ("both", "pytorch"):
        # Measure PT init+first-run
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_pt(x)
        t_pt_init = time.perf_counter() - t0
        t0 = time.time()
        for _ in range(args.iters):
            with torch.no_grad():
                _ = model_pt(x)
        t_pt = (time.time() - t0) / args.iters

    t_tt = None
    if args.backend in ("both", "ttnn"):
        # Measure TTNN init+first-run (includes kernel compile/prep)
        with torch.no_grad():
            _ = model_tt(x)
        t_tt_init = time.perf_counter() - t_init_start
        t0 = time.time()
        for _ in range(args.iters):
            with torch.no_grad():
                _ = model_tt(x)
        t_tt = (time.time() - t0) / args.iters

    if t_pt is not None and t_tt is not None:
        print(f"Init + First Run: PT {t_pt_init*1000:.2f} ms | TTNN {t_tt_init*1000:.2f} ms")
        print(f"Full Model: PT {t_pt*1000:.2f} ms | TTNN {t_tt*1000:.2f} ms | Speedup x{t_pt/max(t_tt,1e-6):.2f}")
    elif t_pt is not None:
        print(f"Init + First Run: PT {t_pt_init*1000:.2f} ms")
        print(f"Full Model: PT {t_pt*1000:.2f} ms")
    elif t_tt is not None:
        print(f"Init + First Run: TTNN {t_tt_init*1000:.2f} ms")
        print(f"Full Model: TTNN {t_tt*1000:.2f} ms")

    try:
        model_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
