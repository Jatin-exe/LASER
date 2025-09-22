#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from ttnn_impl.full_dfine_ttnn_model import DFINE_TTNN


def to_tensor(img_path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    tfm = T.Compose([T.Resize(size), T.ToTensor()])
    return tfm(img).unsqueeze(0)


def metrics(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def main():
    p = argparse.ArgumentParser("Compare full DFINE model: PyTorch vs TTNN")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--max_images", type=int, default=3)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--mae_tol", type=float, default=5e-2)
    p.add_argument("--max_tol", type=float, default=2e-1)
    p.add_argument("--cos_tol", type=float, default=0.999)
    args = p.parse_args()

    cfg = YAMLConfig(args.config_path, resume=args.model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model_pt = cfg.model.eval()
    try:
        if hasattr(model_pt, 'encoder'):
            model_pt.encoder.eval_spatial_size = None
        if hasattr(model_pt, 'decoder'):
            model_pt.decoder.eval_spatial_size = None
    except Exception:
        pass

    model_tt = DFINE_TTNN(model_pt, device_id=args.device_id)

    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    img_paths = img_paths[: args.max_images]
    if not img_paths:
        print("No images found in", args.images_dir)
        return

    for idx, img_path in enumerate(img_paths):
        x = to_tensor(img_path, (args.size, args.size))
        with torch.no_grad():
            out_pt = model_pt(x)
            out_tt = model_tt(x)

        mae_l, mx_l, cos_l = metrics(out_pt["pred_logits"], out_tt["pred_logits"]) \
            if "pred_logits" in out_pt else (0.0, 0.0, 1.0)
        mae_b, mx_b, cos_b = metrics(out_pt["pred_boxes"], out_tt["pred_boxes"])
        s_l = "OK" if (mae_l <= args.mae_tol and mx_l <= args.max_tol and cos_l >= args.cos_tol) else "FAIL"
        s_b = "OK" if (mae_b <= args.mae_tol and mx_b <= args.max_tol and cos_b >= args.cos_tol) else "FAIL"
        print(f"Image {idx}: {img_path.name} final: logits {s_l} boxes {s_b}")
        print(f"  logits: mae={mae_l:.4e} max={mx_l:.4e} cos={cos_l:.6f}")
        print(f"  boxes : mae={mae_b:.4e} max={mx_b:.4e} cos={cos_b:.6f}")

    try:
        model_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

