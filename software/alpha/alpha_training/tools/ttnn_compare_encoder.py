#!/usr/bin/env python3
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

# Ensure project root and tools/ are importable regardless of CWD
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from src.core import YAMLConfig


def metrics(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def forward_pt_encoder_debug(encoder, feats: Sequence[torch.Tensor]):
    """Run PyTorch encoder and return stage-wise outputs for comparison.

    Returns: (outs, debug)
      - outs: final outputs (PAN outputs)
      - debug: dict with keys: proj, encoder, fpn, pan, final
    """
    debug: Dict[str, object] = {}

    # Projections
    proj_feats = [encoder.input_proj[i](feat) for i, feat in enumerate(feats)]
    debug["proj"] = proj_feats

    # Encoder transformer layers
    if encoder.num_encoder_layers > 0:
        enc_updated = list(proj_feats)
        for i, enc_ind in enumerate(encoder.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
            if encoder.training or encoder.eval_spatial_size is None:
                pos_embed = encoder.build_2d_sincos_position_embedding(
                    w, h, encoder.hidden_dim, encoder.pe_temperature
                ).to(src_flatten.device)
            else:
                pos_embed = getattr(encoder, f"pos_embed{enc_ind}", None).to(src_flatten.device)

            memory: torch.Tensor = encoder.encoder[i](src_flatten, pos_embed=pos_embed)
            enc_updated[enc_ind] = (
                memory.permute(0, 2, 1).reshape(-1, encoder.hidden_dim, h, w).contiguous()
            )
        debug["encoder"] = enc_updated
    else:
        debug["encoder"] = proj_feats

    # FPN top-down
    inner_outs: List[torch.Tensor] = [debug["encoder"][-1]]
    for idx in range(len(encoder.in_channels) - 1, 0, -1):
        feat_heigh = inner_outs[0]
        feat_low = debug["encoder"][idx - 1]
        feat_heigh = encoder.lateral_convs[len(encoder.in_channels) - 1 - idx](feat_heigh)
        inner_outs[0] = feat_heigh
        upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
        inner_out = encoder.fpn_blocks[len(encoder.in_channels) - 1 - idx](
            torch.concat([upsample_feat, feat_low], dim=1)
        )
        inner_outs.insert(0, inner_out)
    debug["fpn"] = inner_outs

    # PAN bottom-up
    outs = [inner_outs[0]]
    for idx in range(len(encoder.in_channels) - 1):
        feat_low = outs[-1]
        feat_height = inner_outs[idx + 1]
        downsample_feat = encoder.downsample_convs[idx](feat_low)
        out = encoder.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
        outs.append(out)
    debug["pan"] = outs
    debug["final"] = outs
    return outs, debug


def main():
    p = argparse.ArgumentParser("Compare HybridEncoder: PyTorch vs TTNN (manual)")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--max_images", type=int, default=3)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--size", type=int, default=512, help="Resize square side for inputs")
    p.add_argument("--mae_tol", type=float, default=5e-2)
    p.add_argument("--max_tol", type=float, default=2.0e-1)
    p.add_argument("--cos_tol", type=float, default=0.999)
    p.add_argument("--compare_stage", choices=["proj", "encoder", "fpn", "pan", "final"], default="proj")
    p.add_argument("--dump_dir", type=Path, default=None, help="Optional directory to dump stage features")
    p.add_argument("--summary_only", action="store_true", help="Skip per-image details, show summary only")
    args = p.parse_args()

    # Build DFINE and load checkpoint
    cfg = YAMLConfig(args.config_path, resume=args.model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.eval()
    # Force dynamic pos_embed size for arbitrary image sizes
    try:
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
            model.encoder.eval_spatial_size = None
    except Exception:
        pass

    backbone_pt = model.backbone.eval()
    encoder_pt = model.encoder.eval()

    # Manual TTNN backbone to generate realistic encoder inputs
    from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual

    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=args.device_id)

    # TTNN HybridEncoder (share device with backbone for realism)
    from ttnn_impl.hybrid_encoder_ttnn import HybridEncoderTTNN

    encoder_tt = HybridEncoderTTNN(encoder_pt, device=backbone_tt.device)

    # Collect images
    from tools.ttnn_compare_backbone import to_tensor

    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths.extend(sorted(Path(args.images_dir).glob(ext)))
    img_paths = img_paths[: args.max_images]
    if not img_paths:
        print("No images found in", args.images_dir)
        return

    # Aggregated stats
    step_stats: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_paths):
        x = to_tensor(img_path, size=(args.size, args.size))

        # Get backbone features from TTNN path (torch tensors)
        feats_tt = backbone_tt(x)
        # Ensure PyTorch baseline receives float tensors
        feats_pt = [t.float() for t in feats_tt]

        # Run PyTorch encoder and TTNN encoder with the same inputs
        with torch.no_grad():
            pt_outs, pt_debug = forward_pt_encoder_debug(encoder_pt, feats_pt)
            tt_outs, tt_debug = encoder_tt.forward_debug(feats_tt)

        # Decide which stage to compare
        def get_stage_tensors(tag: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            if tag == "final" or tag == "pan":
                return pt_outs, tt_outs
            if tag not in pt_debug:
                raise KeyError(f"PT debug missing stage '{tag}'")
            if tag not in tt_debug:
                # Not implemented yet on TTNN; skip comparison gracefully
                return pt_debug[tag], []
            return pt_debug[tag], tt_debug[tag]

        pt_list, tt_list = get_stage_tensors(args.compare_stage)

        if not args.summary_only:
            print(f"Image {idx}: {img_path.name} stage={args.compare_stage}")

        if not tt_list:
            print(f"  TTNN stage '{args.compare_stage}' not implemented yet; skipping comparisons.")
        else:
            # Compare feature-by-feature for the selected stage
            for j, (pt_t, tt_t) in enumerate(zip(pt_list, tt_list)):
                mae, mx, cos = metrics(pt_t, tt_t)
                step_stats[args.compare_stage].append((mae, mx, cos))
                status = "OK" if (mae <= args.mae_tol and mx <= args.max_tol and cos >= args.cos_tol) else "FAIL"
                if not args.summary_only:
                    print(
                        f"  feat[{j}]: shape={tuple(pt_t.shape)} mae={mae:.4e} max={mx:.4e} cos={cos:.6f} => {status}"
                    )

        # Optionally dump tensors for deeper inspection
        if args.dump_dir:
            dump_base = args.dump_dir / f"{idx:03d}_{img_path.stem}_{args.compare_stage}"
            dump_base.mkdir(exist_ok=True)
            for j, t in enumerate(pt_list):
                torch.save(t.cpu(), dump_base / f"pt_stage_{args.compare_stage}_{j}.pt")
            for j, t in enumerate(tt_list):
                torch.save(t.cpu(), dump_base / f"tt_stage_{args.compare_stage}_{j}.pt")

    # Cleanup TTNN handles
    try:
        encoder_tt.close()
    except Exception:
        pass
    try:
        backbone_tt.close()
    except Exception:
        pass

    # Summary
    def summarize(entries: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if not entries:
            return 0.0, 0.0, 1.0
        maes = [x[0] for x in entries]
        mxs = [x[1] for x in entries]
        coss = [x[2] for x in entries]
        return max(maes), max(mxs), min(coss)

    print("\n=== Encoder Stage Summary (max MAE / max abs / min cos) ===")
    for tag in ["proj", "encoder", "fpn", "pan", "final"]:
        mae, mx, cos = summarize(step_stats.get(tag, []))
        print(f"  {tag:7s}: mae<= {mae:.4e} max<= {mx:.4e} cos>= {cos:.6f}")


if __name__ == "__main__":
    main()
