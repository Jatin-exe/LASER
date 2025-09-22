import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttnn_impl.mha_ttnn import TTNNMHA


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def test_ttnn_mha_matches_torch():
    torch.manual_seed(0)
    B, S, E, H = 1, 64, 256, 8
    x = torch.randn(B, S, E)
    mha_pt = torch.nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)
    mha_pt.eval()

    # PyTorch reference
    with torch.no_grad():
        y_pt, _ = mha_pt(x, x, x, need_weights=False)

    # TTNN version
    device = ttnn.open_device(device_id=0)
    try:
        enc_tt = TTNNMHA(mha_pt, device=device)
        with torch.no_grad():
            y_tt = enc_tt(x)
    finally:
        ttnn.close_device(device)

    mae, mx, cos = _metrics(y_pt, y_tt)
    assert mae <= 5e-2
    assert mx <= 2e-1
    assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)


def test_ttnn_mha_small_head_dim():
    # Validate fallback path when head_dim < 32 (e.g., E=128, H=8 => Dh=16)
    torch.manual_seed(0)
    B, S, E, H = 1, 64, 128, 8
    x = torch.randn(B, S, E)
    mha_pt = torch.nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)
    mha_pt.eval()

    with torch.no_grad():
        y_pt, _ = mha_pt(x, x, x, need_weights=False)

    device = ttnn.open_device(device_id=0)
    try:
        enc_tt = TTNNMHA(mha_pt, device=device)
        with torch.no_grad():
            y_tt = enc_tt(x)
    finally:
        ttnn.close_device(device)

    mae, mx, cos = _metrics(y_pt, y_tt)
    assert mae <= 5e-2
    assert mx <= 2e-1
    assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
