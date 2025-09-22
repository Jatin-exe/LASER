import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from D_FINE.src.zoo.dfine.hybrid_encoder import TransformerEncoderLayer

try:
from ttnn_impl.hybrid_encoder_ttnn import _TTNNConfig, _TTNNTransformerEncoderLayer as TTLayer


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def test_ttnn_encoder_layer_matches_torch():
    torch.manual_seed(0)
    B, H, W, C = 1, 4, 4, 256
    S = H * W
    x = torch.randn(B, S, C)
    pos = torch.randn(1, S, C)
    pt = TransformerEncoderLayer(d_model=C, nhead=8, dim_feedforward=512, dropout=0.0, activation="gelu", normalize_before=False).eval()
    device = ttnn.open_device(device_id=0)
    try:
        tt = TTLayer(pt, device, _TTNNConfig(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        with torch.no_grad():
            y_pt = pt(x, pos_embed=pos)
            y_tt = tt(x, pos_embed=pos)
        mae, mx, cos = _metrics(y_pt, y_tt)
        assert mae <= 5e-2
        assert mx <= 2e-1
        assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
    finally:
        ttnn.close_device(device)
