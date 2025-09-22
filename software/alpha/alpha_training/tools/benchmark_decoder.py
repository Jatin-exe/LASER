#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from tools.ttnn_compare_decoder import to_tensor


def build_model(cfg_path: str, model_path: str):
    cfg = YAMLConfig(cfg_path, resume=model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    return cfg.model.eval()


def main():
    p = argparse.ArgumentParser("Benchmark DFINE Decoder pipeline: PyTorch vs TTNN")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--backend", choices=["both", "pytorch", "ttnn"], default="both")
    args = p.parse_args()

    model = build_model(args.config_path, args.model_path)
    backbone_pt = model.backbone.eval()
    encoder_pt = model.encoder.eval()
    decoder_pt = model.decoder.eval()

    from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual
    from ttnn_impl.hybrid_encoder_ttnn import HybridEncoderTTNN
    from ttnn_impl.dfine_decoder_ttnn import DFINETransformerTTNN

    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=args.device_id)
    encoder_tt = HybridEncoderTTNN(encoder_pt, device=backbone_tt.device, return_stage="final")
    decoder_tt = DFINETransformerTTNN(decoder_pt, device=backbone_tt.device)

    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    if not img_paths:
        print("No images found in", args.images_dir)
        return

    x = to_tensor(img_paths[0], (args.size, args.size))

    t_pt = None
    if args.backend in ("both", "pytorch"):
        for _ in range(args.warmup):
            with torch.no_grad():
                f = backbone_pt(x)
                f = encoder_pt(f)
                _ = decoder_pt(f)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(args.iters):
            with torch.no_grad():
                f = backbone_pt(x)
                f = encoder_pt(f)
                _ = decoder_pt(f)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_pt = (time.time() - t0) / args.iters

    t_tt = None
    if args.backend in ("both", "ttnn"):
        for _ in range(args.warmup):
            with torch.no_grad():
                f = backbone_tt(x)
                f = encoder_tt(f)
                _ = decoder_tt(f)
        t0 = time.time()
        for _ in range(args.iters):
            with torch.no_grad():
                f = backbone_tt(x)
                f = encoder_tt(f)
                _ = decoder_tt(f)
        t_tt = (time.time() - t0) / args.iters

    if t_pt is not None and t_tt is not None:
        print(f"PT avg latency: {t_pt*1000:.2f} ms | TTNN avg latency: {t_tt*1000:.2f} ms | Speedup x{t_pt/max(t_tt,1e-6):.2f}")
    elif t_pt is not None:
        print(f"PT avg latency: {t_pt*1000:.2f} ms")
    elif t_tt is not None:
        print(f"TTNN avg latency: {t_tt*1000:.2f} ms")

    try:
        decoder_tt.close(); encoder_tt.close(); backbone_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
