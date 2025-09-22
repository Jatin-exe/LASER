import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttnn_impl.dfine_decoder_ttnn import TTNNLinearWrap, TTNNMLPWrap


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def test_ttnn_linear_mlp_heads_equivalence():
    torch.manual_seed(0)
    # Build small linear and MLP
    lin = torch.nn.Linear(16, 7, bias=True).eval()
    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_layers = 2
            self.layers = torch.nn.ModuleList([torch.nn.Linear(16, 16), torch.nn.Linear(16, 4)])
            self.act = torch.nn.GELU()
        def forward(self, x):
            x = self.act(self.layers[0](x))
            x = self.layers[1](x)
            return x
    mlp = _MLP().eval()

    # Reuse wrappers from decoder module
    device = ttnn.open_device(device_id=0)
    try:
        lin_tt = TTNNLinearWrap(lin, device, ttnn.bfloat16, ttnn)
        mlp_tt = TTNNMLPWrap(mlp, device, ttnn.bfloat16, ttnn)

        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            y_lin_pt = lin(x)
            y_lin_tt = lin_tt(x)
            y_mlp_pt = mlp(x)
            y_mlp_tt = mlp_tt(x)

        mae_l, mx_l, cos_l = _metrics(y_lin_pt, y_lin_tt)
        mae_m, mx_m, cos_m = _metrics(y_mlp_pt, y_mlp_tt)
        assert mae_l <= 5e-2 and cos_l >= 0.99
        assert mae_m <= 5e-2 and cos_m >= 0.99
    finally:
        ttnn.close_device(device)
