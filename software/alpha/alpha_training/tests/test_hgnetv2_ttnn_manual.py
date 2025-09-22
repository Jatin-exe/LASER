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
from ttnn_impl.hgnetv2_ttnn_manual import HGNetv2TTNNManual


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


@pytest.mark.parametrize("arch", ["B0"])
def test_stage_outputs_close(arch):
    torch.manual_seed(0)
    backbone_pt = HGNetv2(arch, pretrained=False).eval()
    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=0)

    try:
        x = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            pt_stages = [backbone_pt.stem(x)]
            y = pt_stages[-1]
            for stage in backbone_pt.stages:
                y = stage(y)
                pt_stages.append(y)

        tt_stages = backbone_tt.collect_stage_outputs(x, include_stem=True)

        assert len(tt_stages) == len(pt_stages)
        for idx, (pt_tensor, tt_tensor) in enumerate(zip(pt_stages, tt_stages)):
            mae, mx, cos = _metrics(pt_tensor, tt_tensor)
            assert mae <= 1e-2, f"Stage {idx} MAE too high: {mae}"
            assert mx <= 5e-2, f"Stage {idx} MAX too high: {mx}"
            assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
    finally:
        backbone_tt.close()


def test_forward_matches_return_idx():
    torch.manual_seed(1)
    backbone_pt = HGNetv2("B0", pretrained=False).eval()
    backbone_tt = HGNetv2TTNNManual(backbone_pt, device_id=0)

    try:
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            pt_feats = backbone_pt(x)
            tt_feats = backbone_tt(x)

        assert len(pt_feats) == len(tt_feats)
        for idx, (pt_tensor, tt_tensor) in enumerate(zip(pt_feats, tt_feats)):
            mae, mx, cos = _metrics(pt_tensor, tt_tensor)
            assert mae <= 1e-2
            assert mx <= 5e-2
            assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)
    finally:
        backbone_tt.close()
