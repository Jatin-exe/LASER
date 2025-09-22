#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Make project root importable regardless of CWD
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig


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


def build_model(cfg_path: str, model_path: str):
    cfg = YAMLConfig(cfg_path, resume=model_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.eval()
    try:
        if hasattr(model, 'encoder'):
            model.encoder.eval_spatial_size = None
        if hasattr(model, 'decoder'):
            model.decoder.eval_spatial_size = None
    except Exception:
        pass
    return model


def main():
    p = argparse.ArgumentParser("Compare DFINE Transformer Decoder: PyTorch vs TTNN")
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--max_images", type=int, default=3)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--mae_tol", type=float, default=5e-2)
    p.add_argument("--max_tol", type=float, default=2e-1)
    p.add_argument("--cos_tol", type=float, default=0.999)
    p.add_argument("--compare_stage", choices=["enc_init", "final"], default="enc_init")
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
    img_paths = img_paths[: args.max_images]
    if not img_paths:
        print("No images found in", args.images_dir)
        return

    for idx, img_path in enumerate(img_paths):
        x = to_tensor(img_path, (args.size, args.size))
        with torch.no_grad():
            feats_tt = backbone_tt(x)
            feats_tt = encoder_tt(feats_tt)

            # PT reference: build memory, enc_output, enc_scores, topk selection
            memory_pt, spatial_shapes = decoder_pt._get_encoder_input(feats_tt)
            enc_out_pt = decoder_pt.enc_output(memory_pt)
            enc_scores_pt = decoder_pt.enc_score_head(enc_out_pt)
            if enc_scores_pt.shape[-1] == 1:
                scores_pt = enc_scores_pt.squeeze(-1)
            else:
                scores_pt = enc_scores_pt.max(dim=-1).values
            anchors_pt, _ = decoder_pt._generate_anchors(spatial_shapes, device=memory_pt.device)
            if memory_pt.shape[0] > 1:
                anchors_pt = anchors_pt.repeat(memory_pt.shape[0], 1, 1)
            topk_scores_pt, topk_ind_pt = torch.topk(scores_pt, decoder_pt.num_queries, dim=1)
            topk_memory_pt = memory_pt.gather(1, topk_ind_pt.unsqueeze(-1).repeat(1, 1, memory_pt.shape[-1]))
            topk_anchors_pt = anchors_pt.gather(1, topk_ind_pt.unsqueeze(-1).repeat(1, 1, anchors_pt.shape[-1]))
            enc_topk_bbox_unact_pt = decoder_pt.enc_bbox_head(topk_memory_pt) + topk_anchors_pt

            # TTNN path (Step 3.1)
            out_tt, debug_tt = decoder_tt.forward_debug(feats_tt)

        if args.compare_stage == "enc_init":
            mae1, mx1, cos1 = metrics(topk_scores_pt, debug_tt["topk_scores"]) if enc_scores_pt.shape[-1] == 1 else metrics(topk_ind_pt.float(), debug_tt["topk_ind"].float())
            mae2, mx2, cos2 = metrics(enc_topk_bbox_unact_pt, out_tt["enc_topk_bbox_unact"])
            status1 = "OK" if (mae1 <= args.mae_tol and mx1 <= args.max_tol and cos1 >= args.cos_tol) else "FAIL"
            status2 = "OK" if (mae2 <= args.mae_tol and mx2 <= args.max_tol and cos2 >= args.cos_tol) else "FAIL"
            print(f"Image {idx}: {img_path.name} enc_init: scores {status1} bbox_unact {status2}")
            print(f"  scores: mae={mae1:.4e} max={mx1:.4e} cos={cos1:.6f}")
            print(f"  bbox_u: mae={mae2:.4e} max={mx2:.4e} cos={cos2:.6f}")
        else:
            # Full pipeline compare
            out_pt = decoder_pt(feats_tt)
            out_tt_full = decoder_tt.forward(feats_tt)

            pt_logits = out_pt["pred_logits"].detach()
            pt_boxes = out_pt["pred_boxes"].detach()
            tt_logits = out_tt_full["pred_logits"].detach()
            tt_boxes = out_tt_full["pred_boxes"].detach()

            mae_l, mx_l, cos_l = metrics(pt_logits, tt_logits)
            mae_b, mx_b, cos_b = metrics(pt_boxes, tt_boxes)
            s_l = "OK" if (mae_l <= args.mae_tol and mx_l <= args.max_tol and cos_l >= args.cos_tol) else "FAIL"
            s_b = "OK" if (mae_b <= args.mae_tol and mx_b <= args.max_tol and cos_b >= args.cos_tol) else "FAIL"
            print(f"Image {idx}: {img_path.name} final: logits {s_l} boxes {s_b}")
            print(f"  logits: mae={mae_l:.4e} max={mx_l:.4e} cos={cos_l:.6f}")
            print(f"  boxes : mae={mae_b:.4e} max={mx_b:.4e} cos={cos_b:.6f}")

    try:
        decoder_tt.close()
        encoder_tt.close()
        backbone_tt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
