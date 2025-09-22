import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    # relative PATHS ffs

from D_FINE.src.zoo.dfine.dfine_decoder import MSDeformableAttention
from D_FINE.src.zoo.dfine.dfine_decoder import TransformerDecoderLayer
from D_FINE.src.zoo.dfine.dfine_decoder import TransformerDecoder
from D_FINE.src.zoo.dfine.dfine_decoder import Integral
from D_FINE.src.zoo.dfine.dfine_utils import weighting_function

from ttnn_impl.dfine_decoder_ttnn import TTNNMSDeformableAttention


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a64 = a.detach().float().cpu().reshape(-1)
    b64 = b.detach().float().cpu().reshape(-1)
    mae = torch.mean(torch.abs(a64 - b64)).item()
    mx = torch.max(torch.abs(a64 - b64)).item()
    denom = (torch.norm(a64) * torch.norm(b64)).item()
    cos = (torch.dot(a64, b64).item() / denom) if denom != 0 else 1.0
    return mae, mx, cos


def _random_value_list(B, H, C_head, shapes, device):
    vals = []
    for h, w in shapes:
        vals.append(torch.randn(B, H, C_head, h * w, device=device))
    return vals


def test_msdeformableattention_equivalence_default():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, Lq = 1, 32
    embed_dim, num_heads = 128, 8
    head_dim = embed_dim // num_heads
    num_levels = 2
    num_points = 3
    shapes = [(16, 16), (8, 8)]

    msda_pt = MSDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, method="default").eval()
    # Build inputs
    query = torch.randn(B, Lq, embed_dim, device=device)
    reference_points = torch.rand(B, Lq, num_levels, 2, device=device)
    value = _random_value_list(B, num_heads, head_dim, shapes, device)
    value_spatial_shapes = shapes

    from D_FINE.src.zoo.dfine.utils import deformable_attention_core_func_v2
    with torch.no_grad():
        # Reproduce PyTorch MSDA math directly to avoid signature ambiguity
        sampling_offsets = msda_pt.sampling_offsets(query).reshape(B, Lq, num_heads, sum([num_points]*num_levels), 2)
        attention_weights = torch.softmax(msda_pt.attention_weights(query).reshape(B, Lq, num_heads, sum([num_points]*num_levels)), dim=-1)
        # Build per-level sampling locations (last dim=2 case)
        offset_normalizer = torch.tensor(value_spatial_shapes, dtype=query.dtype, device=query.device)
        offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, num_levels, 1, 2)
        off_splits = sampling_offsets.split([num_points]*num_levels, dim=3)
        sampling_locations_per_level = []
        for lvl, (h, w) in enumerate(value_spatial_shapes):
            loc = reference_points[:, :, None, lvl, None, :2] + off_splits[lvl] / offset_normalizer[:, :, :, lvl, :, :]
            sampling_locations_per_level.append(loc)
        # Flatten levels: [B,Lq,H, sumP, 2]
        sampling_locations = torch.cat(sampling_locations_per_level, dim=3)
        y_pt = deformable_attention_core_func_v2(value, value_spatial_shapes, sampling_locations, attention_weights, [num_points]*num_levels, method="default")

    # TTNN version
    dev = ttnn.open_device(device_id=0)
    try:
        msda_tt = TTNNMSDeformableAttention(msda_pt, device=dev)
        with torch.no_grad():
            y_tt = msda_tt(query, reference_points, value, value_spatial_shapes)
    finally:
        ttnn.close_device(dev)

    mae, mx, cos = _metrics(y_pt, y_tt)
    # BF16 grid sample will have some error; keep tight enough for structure
    assert mae <= 1.0e-2
    assert mx <= 2.0e-1
    assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)


def test_decoder_layer_equivalence_default():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, Lq = 1, 32
    embed_dim, num_heads = 128, 8
    head_dim = embed_dim // num_heads
    num_levels = 2
    num_points = 3
    shapes = [(16, 16), (8, 8)]

    layer_pt = TransformerDecoderLayer(d_model=embed_dim, n_head=num_heads, dim_feedforward=256, dropout=0.0, activation="gelu", n_levels=num_levels, n_points=num_points, cross_attn_method="default").eval()

    target = torch.randn(B, Lq, embed_dim, device=device)
    query_pos = torch.randn(B, Lq, embed_dim, device=device)
    reference_points = torch.rand(B, Lq, num_levels, 2, device=device)
    # Build value list [B,H,C_head, H*W]
    value = _random_value_list(B, num_heads, head_dim, shapes, device)
    spatial_shapes = shapes

    from D_FINE.src.zoo.dfine.utils import deformable_attention_core_func_v2
    with torch.no_grad():
        # Manual PyTorch path to avoid MSDA forward shape ambiguity
        q = k = target + query_pos
        sa, _ = layer_pt.self_attn(q, k, value=target, attn_mask=None)
        x = layer_pt.norm1(target + sa)
        # Cross-attn via core func
        sampling_offsets = layer_pt.cross_attn.sampling_offsets(x).reshape(B, Lq, num_heads, sum([num_points]*num_levels), 2)
        attention_weights = torch.softmax(layer_pt.cross_attn.attention_weights(x).reshape(B, Lq, num_heads, sum([num_points]*num_levels)), dim=-1)
        # Build per-level sampling locations for last dim=2
        offset_normalizer = torch.tensor(spatial_shapes, dtype=x.dtype, device=x.device)
        offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, num_levels, 1, 2)
        off_splits = sampling_offsets.split([num_points]*num_levels, dim=3)
        sampling_locations_per_level = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            loc = reference_points[:, :, None, lvl, None, :2] + off_splits[lvl] / offset_normalizer[:, :, :, lvl, :, :]
            sampling_locations_per_level.append(loc)
        sampling_locations = torch.cat(sampling_locations_per_level, dim=3)
        ca = deformable_attention_core_func_v2(value, spatial_shapes, sampling_locations, attention_weights, [num_points]*num_levels, method="default")
        x = layer_pt.gateway(x, ca)
        # FFN
        x2 = layer_pt.linear2(layer_pt.activation(layer_pt.linear1(x)))
        y_pt = layer_pt.norm3(x + x2)

    # TTNN layer
    dev = ttnn.open_device(device_id=0)
    try:
        from ttnn_impl.dfine_decoder_ttnn import TTNNTransformerDecoderLayer
        layer_tt = TTNNTransformerDecoderLayer(layer_pt, dev)
        with torch.no_grad():
            y_tt = layer_tt(target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=query_pos)
    finally:
        ttnn.close_device(dev)

    mae, mx, cos = _metrics(y_pt, y_tt)
    assert mae <= 1.0e-2
    assert mx <= 2.0e-1
    assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)


def test_integral_equivalence():
    torch.manual_seed(0)
    B, L = 2, 10
    reg_max = 16
    up, reg_scale = torch.tensor([4.0]), 4
    x = torch.randn(B, L, 4 * (reg_max + 1))
    project = weighting_function(reg_max, up, reg_scale, deploy=True)
    integral_pt = Integral(reg_max)
    with torch.no_grad():
        y_pt = integral_pt(x.clone(), project)

    dev = ttnn.open_device(device_id=0)
    try:
        from ttnn_impl.dfine_decoder_ttnn import TTNNIntegral
        integral_tt = TTNNIntegral(reg_max, device=dev)
        with torch.no_grad():
            y_tt = integral_tt(x.clone(), project)
    finally:
        ttnn.close_device(dev)

    mae, mx, cos = _metrics(y_pt, y_tt)
    assert mae <= 7e-2
    assert mx <= 1.2e-1
    assert math.isclose(cos, 1.0, rel_tol=0, abs_tol=1e-3)


def test_on_device_topk_and_gather_selection():
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, L = 2, 256
    hidden_dim = 128
    num_queries = 64

    # Case A: class-agnostic (logits shape [B,L,1])
    enc_logits_a = torch.randn(B, L, 1, device=device)
    anchors = torch.randn(B, L, 4, device=device)
    memory = torch.randn(B, L, hidden_dim, device=device)

    # Torch path
    scores_a = enc_logits_a.squeeze(-1)
    tk_vals_a_pt, tk_idx_a_pt = torch.topk(scores_a, k=num_queries, dim=1)
    topk_anchors_a_pt = anchors.gather(1, tk_idx_a_pt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
    topk_memory_a_pt = memory.gather(1, tk_idx_a_pt.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

    # TTNN path
    dev = ttnn.open_device(device_id=0)
    try:
        scores_tt = ttnn.from_torch(scores_a, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tk_vals_tt, tk_idx_tt = ttnn.topk(scores_tt, k=num_queries, dim=1, largest=True, sorted=True)
        tk_vals_a_tt = ttnn.to_torch(tk_vals_tt)
        tk_idx_a_tt = ttnn.to_torch(tk_idx_tt).to(torch.int64)

        # Prepare tensors transposed so gather can operate on last dim
        anchors_chl = anchors.permute(0, 2, 1).contiguous()  # [B, 4, L]
        memory_chl = memory.permute(0, 2, 1).contiguous()    # [B, C, L]
        anchors_tt = ttnn.from_torch(anchors_chl, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        memory_tt = ttnn.from_torch(memory_chl, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Build gather indices for anchors/memory
        idx_anchor = tk_idx_a_tt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1])
        idx_memory = tk_idx_a_tt.unsqueeze(-1).repeat(1, 1, memory.shape[-1])

        idx_anchor_tt = ttnn.from_torch(idx_anchor.to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        idx_memory_tt = ttnn.from_torch(idx_memory.to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)

        # For gather along last dim (L), transpose index to [B, C, K]
        idx_anchor_chl_tt = ttnn.from_torch(idx_anchor.permute(0, 2, 1).contiguous().to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        idx_memory_chl_tt = ttnn.from_torch(idx_memory.permute(0, 2, 1).contiguous().to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        gathered_anchors = ttnn.gather(anchors_tt, dim=2, index=idx_anchor_chl_tt)  # [B, 4, K]
        gathered_memory = ttnn.gather(memory_tt, dim=2, index=idx_memory_chl_tt)    # [B, C, K]
        topk_anchors_a_tt = ttnn.to_torch(gathered_anchors).permute(0, 2, 1).contiguous()  # [B,K,4]
        topk_memory_a_tt = ttnn.to_torch(gathered_memory).permute(0, 2, 1).contiguous()    # [B,K,C]
    finally:
        ttnn.close_device(dev)

    mae1, mx1, cos1 = _metrics(tk_vals_a_pt, tk_vals_a_tt)
    # Validate gather correctness against TTNN-selected indices on host
    host_anchors_from_ttidx = anchors.gather(1, tk_idx_a_tt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
    host_memory_from_ttidx = memory.gather(1, tk_idx_a_tt.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
    mae2, mx2, cos2 = _metrics(host_anchors_from_ttidx, topk_anchors_a_tt)
    mae3, mx3, cos3 = _metrics(host_memory_from_ttidx, topk_memory_a_tt)

    assert mae1 <= 2.0e-3 and mx1 <= 1.0e-2 and math.isclose(cos1, 1.0, rel_tol=0, abs_tol=1e-5)
    assert mae2 <= 2.0e-3 and mx2 <= 2.0e-2 and math.isclose(cos2, 1.0, rel_tol=0, abs_tol=1e-5)
    assert mae3 <= 2.0e-3 and mx3 <= 2.0e-2 and math.isclose(cos3, 1.0, rel_tol=0, abs_tol=1e-5)

    # Case B: multi-class (logits shape [B,L,C]) -> scores are classwise max
    C_cls = 7
    enc_logits_b = torch.randn(B, L, C_cls, device=device)
    scores_b = enc_logits_b.max(dim=-1).values
    tk_vals_b_pt, tk_idx_b_pt = torch.topk(scores_b, k=num_queries, dim=1)
    topk_anchors_b_pt = anchors.gather(1, tk_idx_b_pt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
    topk_memory_b_pt = memory.gather(1, tk_idx_b_pt.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

    dev = ttnn.open_device(device_id=0)
    try:
        # Compute scores on host for stability, then run on-device topk
        scores_b_tt = ttnn.from_torch(scores_b, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tk_vals_tt2, tk_idx_tt2 = ttnn.topk(scores_b_tt, k=num_queries, dim=1, largest=True, sorted=True)
        tk_vals_b_tt = ttnn.to_torch(tk_vals_tt2)
        tk_idx_b_tt = ttnn.to_torch(tk_idx_tt2).to(torch.int64)

        anchors_chl = anchors.permute(0, 2, 1).contiguous()
        memory_chl = memory.permute(0, 2, 1).contiguous()
        anchors_tt = ttnn.from_torch(anchors_chl, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        memory_tt = ttnn.from_torch(memory_chl, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        idx_anchor = tk_idx_b_tt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1])
        idx_memory = tk_idx_b_tt.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        idx_anchor_tt = ttnn.from_torch(idx_anchor.to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        idx_memory_tt = ttnn.from_torch(idx_memory.to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)

        idx_anchor_chl_tt = ttnn.from_torch(idx_anchor.permute(0, 2, 1).contiguous().to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        idx_memory_chl_tt = ttnn.from_torch(idx_memory.permute(0, 2, 1).contiguous().to(torch.int32), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
        gathered_anchors = ttnn.gather(anchors_tt, dim=2, index=idx_anchor_chl_tt)
        gathered_memory = ttnn.gather(memory_tt, dim=2, index=idx_memory_chl_tt)
        topk_anchors_b_tt = ttnn.to_torch(gathered_anchors).permute(0, 2, 1).contiguous()
        topk_memory_b_tt = ttnn.to_torch(gathered_memory).permute(0, 2, 1).contiguous()
    finally:
        ttnn.close_device(dev)

    mae1, mx1, cos1 = _metrics(tk_vals_b_pt, tk_vals_b_tt)
    host_anchors_from_ttidx = anchors.gather(1, tk_idx_b_tt.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
    host_memory_from_ttidx = memory.gather(1, tk_idx_b_tt.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
    mae2, mx2, cos2 = _metrics(host_anchors_from_ttidx, topk_anchors_b_tt)
    mae3, mx3, cos3 = _metrics(host_memory_from_ttidx, topk_memory_b_tt)

    assert mae1 <= 5.0e-3 and mx1 <= 5.0e-2 and math.isclose(cos1, 1.0, rel_tol=0, abs_tol=1e-4)
    assert mae2 <= 2.0e-3 and mx2 <= 2.0e-2 and math.isclose(cos2, 1.0, rel_tol=0, abs_tol=1e-5)
    assert mae3 <= 2.0e-3 and mx3 <= 2.0e-2 and math.isclose(cos3, 1.0, rel_tol=0, abs_tol=1e-5)
