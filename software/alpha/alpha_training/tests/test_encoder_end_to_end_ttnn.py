import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from D_FINE.src.nn.backbone.hgnetv2 import HGNetv2
from D_FINE.src.zoo.dfine.hybrid_encoder import HybridEncoder

from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual
from ttnn_impl.hybrid_encoder_ttnn import HybridEncoderTTNN


# accuracy how close do we need this here ? 
def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def test_encoder_end_to_end_matches_pytorch():
    torch.manual_seed(0)
    backbone_pt = HGNetv2("B0", pretrained=False).eval()
    encoder_pt = HybridEncoder(in_channels=[256, 512, 1024]).eval()
    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=0)
    encoder_tt = HybridEncoderTTNN(encoder_pt, device=backbone_tt.device)

    try:
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            pt_feats = backbone_pt(x)
            # PyTorch stage breakdown
            # 1) projections
            proj_pt = [encoder_pt.input_proj[i](f) for i, f in enumerate(pt_feats)]
            # 2) encoder transformer
            enc_pt_feats = list(proj_pt)
            if encoder_pt.num_encoder_layers > 0:
                for i, enc_ind in enumerate(encoder_pt.use_encoder_idx):
                    h, w = proj_pt[enc_ind].shape[2:]
                    src_flat = proj_pt[enc_ind].flatten(2).permute(0, 2, 1)
                    pos_embed = encoder_pt.build_2d_sincos_position_embedding(
                        w, h, encoder_pt.hidden_dim, encoder_pt.pe_temperature
                    ).to(src_flat.device)
                    mem = encoder_pt.encoder[i](src_flat, pos_embed=pos_embed)
                    enc_pt_feats[enc_ind] = mem.permute(0, 2, 1).reshape(-1, encoder_pt.hidden_dim, h, w).contiguous()
            # 3) FPN
            inner_outs_pt = [enc_pt_feats[-1]]
            for idx in range(len(encoder_pt.in_channels) - 1, 0, -1):
                feat_heigh = inner_outs_pt[0]
                feat_low = enc_pt_feats[idx - 1]
                feat_heigh = encoder_pt.lateral_convs[len(encoder_pt.in_channels) - 1 - idx](feat_heigh)
                inner_outs_pt[0] = feat_heigh
                upsample_feat = torch.nn.functional.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
                inner_out = encoder_pt.fpn_blocks[len(encoder_pt.in_channels) - 1 - idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
                inner_outs_pt.insert(0, inner_out)
            # 4) PAN
            outs_pt = [inner_outs_pt[0]]
            for idx in range(len(encoder_pt.in_channels) - 1):
                feat_low = outs_pt[-1]
                feat_height = inner_outs_pt[idx + 1]
                downsample_feat = encoder_pt.downsample_convs[idx](feat_low)
                out = encoder_pt.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
                outs_pt.append(out)

            feats_tt = backbone_tt(x)
            outs_tt, dbg_tt = encoder_tt.forward_debug(feats_tt)

        # Compare by stage
        for stage_name, pt_list, tt_list in [
            ("proj", proj_pt, dbg_tt["proj"]),
            ("encoder", enc_pt_feats, dbg_tt["encoder"]),
            ("fpn", inner_outs_pt, dbg_tt["fpn"]),
            ("pan", outs_pt, dbg_tt["pan"]),
            ("final", outs_pt, outs_tt),
        ]:
            assert len(pt_list) == len(tt_list), f"Stage {stage_name} length mismatch"
            for i, (a, b) in enumerate(zip(pt_list, tt_list)):
                mae, mx, cos = _metrics(a, b)
                assert mae <= 5e-2, f"{stage_name}[{i}] MAE too high: {mae}"
                assert mx <= 2e-1, f"{stage_name}[{i}] MAX too high: {mx}"
                assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
    finally:
        try:
            encoder_tt.close()
        except Exception:
            pass
        try:
            backbone_tt.close()
        except Exception:
            pass
