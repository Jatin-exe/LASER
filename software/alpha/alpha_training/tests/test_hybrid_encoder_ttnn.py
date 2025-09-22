import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from D_FINE.src.zoo.dfine.hybrid_encoder import HybridEncoder
from ttnn_impl.hybrid_encoder_ttnn import HybridEncoderTTNN


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def _rand_feats():
    # Create small, stride-consistent feature maps
    torch.manual_seed(0)
    f3 = torch.randn(1, 512, 20, 20)
    f4 = torch.randn(1, 1024, 10, 10)
    f5 = torch.randn(1, 2048, 5, 5)
    return [f3, f4, f5]


def test_input_projection_matches_pytorch():
    torch.manual_seed(0)
    enc_pt = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        num_encoder_layers=1,
    ).eval()
    enc_tt = HybridEncoderTTNN(enc_pt, device_id=0)

    feats = _rand_feats()

    try:
        with torch.no_grad():
            proj_pt = [enc_pt.input_proj[i](t) for i, t in enumerate(feats)]
            proj_tt = enc_tt(feats)

        assert len(proj_pt) == len(proj_tt)
        for i, (a, b) in enumerate(zip(proj_pt, proj_tt)):
            mae, mx, cos = _metrics(a, b)
            assert mae <= 2.0e-2, f"proj[{i}] MAE too high: {mae}"
            # Allow slightly looser max tolerance for isolated conv bring-up
            assert mx <= 1.0e-1, f"proj[{i}] MAX too high: {mx}"
            assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
    finally:
        enc_tt.close()
