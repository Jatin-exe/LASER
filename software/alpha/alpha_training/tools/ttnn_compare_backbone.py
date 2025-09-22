#!/usr/bin/env python3
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Ensure project root and tools/ are importable regardless of CWD
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from src.core import YAMLConfig


def to_tensor(img_path: Path, size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
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


def collect_pt_stage_outputs(
    backbone: torch.nn.Module, x: torch.Tensor, include_stem: bool = True
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        y = backbone.stem(x)
        if include_stem:
            outputs.append(y)
        for stage in backbone.stages:
            y = stage(y)
            outputs.append(y)
    return outputs


def main():
    p = argparse.ArgumentParser("Compare HGNetv2 backbone: PyTorch vs TTNN (tt-torch)")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--max_images", type=int, default=5)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--size", type=int, default=512, help="Resize square side for inputs")
    p.add_argument("--mae_tol", type=float, default=1e-2)
    p.add_argument("--max_tol", type=float, default=5e-2)
    p.add_argument("--cos_tol", type=float, default=0.999)
    p.add_argument("--dump_dir", type=Path, default=None, help="Optional directory to dump stage features")
    p.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip per-image logs and only print final summary",
    )
    args = p.parse_args()

    # Build DFINE and load checkpoint
    cfg = YAMLConfig(args.config_path, resume=args.model_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.eval()

    # Extract PyTorch backbone
    backbone_pt = model.backbone.eval()

    # Build manual TTNN backbone (full TTNN backbone)
    from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual

    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=args.device_id)

    # Collect images
    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    img_paths = img_paths[: args.max_images]

    if not img_paths:
        print("No images found in", args.images_dir)
        return

    stage_names = ["stem"] + [f"stage{i+1}" for i in range(len(backbone_pt.stages))]
    stage_stats: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    feat_stats: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_paths):
        x = to_tensor(img_path, size=(args.size, args.size))
        pt_stages = collect_pt_stage_outputs(backbone_pt, x, include_stem=True)
        tt_stages = backbone_tt.collect_stage_outputs(x, include_stem=True)

        if not args.summary_only:
            print(f"Image {idx}: {img_path.name}")

        for stage_id, (name, pt_tensor, tt_tensor) in enumerate(zip(stage_names, pt_stages, tt_stages)):
            mae, mx, cos = metrics(pt_tensor, tt_tensor)
            stage_stats[name].append((mae, mx, cos))
            stage_idx = stage_id - 1  # stem => -1, stage0 => 0
            is_return = stage_idx in backbone_pt.return_idx if stage_idx >= 0 else False
            status = "OK" if (mae <= args.mae_tol and mx <= args.max_tol and cos >= args.cos_tol) else "FAIL"
            if not args.summary_only:
                marker = "*" if is_return else " "
                print(
                    f"  {name}{marker}: shape={tuple(pt_tensor.shape)} "
                    f"mae={mae:.4e} max={mx:.4e} cos={cos:.6f} => {status}"
                )
                if status == "FAIL":
                    print("    Hint: verify BN folding / padding / layout for this stage.")

            if is_return:
                feat_stats[stage_idx].append((mae, mx, cos))

        if args.dump_dir:
            dump_base = args.dump_dir / f"{idx:03d}_{img_path.stem}"
            dump_base.mkdir(exist_ok=True)
            for name, pt_tensor, tt_tensor in zip(stage_names, pt_stages, tt_stages):
                torch.save(pt_tensor.cpu(), dump_base / f"pt_{name}.pt")
                torch.save(tt_tensor.cpu(), dump_base / f"tt_{name}.pt")

    try:
        backbone_tt.close()
    except Exception:
        pass

    def summarize(entries: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if not entries:
            return 0.0, 0.0, 1.0
        maes = [x[0] for x in entries]
        mxs = [x[1] for x in entries]
        coss = [x[2] for x in entries]
        return max(maes), max(mxs), min(coss)

    print("\n=== Stage summary (max MAE / max abs / min cos) ===")
    for name in stage_names:
        mae, mx, cos = summarize(stage_stats[name])
        marker = "*" if (name != "stem" and (stage_names.index(name) - 1) in backbone_pt.return_idx) else " "
        print(f"  {name}{marker}: mae<= {mae:.4e} max<= {mx:.4e} cos>= {cos:.6f}")

    print("\n=== Feature summary (return_idx) ===")
    for idx in backbone_pt.return_idx:
        mae, mx, cos = summarize(feat_stats[idx])
        print(f"  stage{idx+1}: mae<= {mae:.4e} max<= {mx:.4e} cos>= {cos:.6f}")


if __name__ == "__main__":
    main()
