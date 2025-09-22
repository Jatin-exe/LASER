"""TTNN port of the D-FINE transformer decoder stack."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import ttnn


from src.zoo.dfine.dfine_utils import weighting_function, distance2bbox
from src.zoo.dfine.utils import inverse_sigmoid

class TTNNLinearWrap(nn.Module):
    def __init__(self, linear_pt: nn.Linear, device, dtype, ttnn_mod):
        super().__init__()
        W = linear_pt.weight.detach().t().contiguous()
        b = linear_pt.bias.detach().reshape(1, 1, -1)
        self.ttnn = ttnn_mod
        self.device = device
        self.dtype = dtype
        self.W = ttnn_mod.from_torch(W, device=device, dtype=dtype, layout=ttnn_mod.TILE_LAYOUT)
        self.b = ttnn_mod.from_torch(b, device=device, dtype=dtype, layout=ttnn_mod.TILE_LAYOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tt = self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
        y_tt = self.ttnn.linear(x_tt, self.W, bias=self.b)
        return self.ttnn.to_torch(y_tt)


class TTNNMLPWrap(nn.Module):
    def __init__(self, mlp_pt: nn.Module, device, dtype, ttnn_mod):
        super().__init__()
        self.ttnn = ttnn_mod
        self.device = device
        self.dtype = dtype
        layers = []
        activations: List[Optional[str]] = []
        # MLP has attributes: num_layers, layers (ModuleList[Linear]), act
        for i, layer in enumerate(mlp_pt.layers):
            layers.append(TTNNLinearWrap(layer, device, dtype, ttnn_mod))
            act_kind = "gelu" if isinstance(mlp_pt.act, nn.GELU) else ("relu" if i < mlp_pt.num_layers - 1 else None)
            activations.append(act_kind)
        activations[-1] = None
        self.layers = nn.ModuleList(layers)
        self.activations = activations

    def _apply_act(self, x_tt, kind: Optional[str]):
        if not kind:
            return x_tt
        if kind == "gelu":
            return self.ttnn.gelu(x_tt)
        return self.ttnn.relu(x_tt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tt = self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
        for i, lin in enumerate(self.layers):
            y_tt = self.ttnn.linear(x_tt, lin.W, bias=lin.b)
            y_tt = self._apply_act(y_tt, self.activations[i])
            x_tt = y_tt
        return self.ttnn.to_torch(x_tt)


class DFINETransformerTTNN(nn.Module):
    def __init__(self, decoder_pt: nn.Module, device=None, device_id: int = 0):
        super().__init__()
        self.ttnn = ttnn
        self._owns_device = device is None
        if device is None:
            try:
                device = ttnn.open_device(device_id=device_id, l1_small_size=655360)
            except TypeError:
                device = ttnn.open_device(device_id=device_id)
        self.device = device
        self.dtype = ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT

        # Keep reference to PyTorch decoder for complex pieces during bring-up
        self.decoder_pt = decoder_pt

        # enc_output: Sequential([Linear, LayerNorm])
        enc_proj = self.decoder_pt.enc_output[0]
        enc_norm = self.decoder_pt.enc_output[1]
        self.W_enc = ttnn.from_torch(
            enc_proj.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.b_enc = ttnn.from_torch(
            enc_proj.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.ln_w = ttnn.from_torch(
            enc_norm.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.ln_b = ttnn.from_torch(
            enc_norm.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )

        # enc_score_head: Linear(hidden_dim -> num_classes or 1)
        self.W_enc_score = ttnn.from_torch(
            self.decoder_pt.enc_score_head.weight.detach().t().contiguous(),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        self.b_enc_score = ttnn.from_torch(
            self.decoder_pt.enc_score_head.bias.detach().reshape(1, 1, -1),
            device=device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
        )

        # Build TTNN decoder layers mapped from PyTorch
        self.layers: List[TTNNTransformerDecoderLayer] = []
        for layer_pt in self.decoder_pt.decoder.layers:
            self.layers.append(TTNNTransformerDecoderLayer(layer_pt, device=self.device, dtype=self.dtype, layout=self.layout))

        # TTNN Integral for distribution-to-distance conversion
        self.integral = TTNNIntegral(int(self.decoder_pt.reg_max), device=self.device, dtype=self.dtype, layout=self.layout)

        # Build TTNN versions of heads (enc/dec score & bbox, pre_bbox, query_pos)
        def build_tt_linear(linear: nn.Linear):
            W = linear.weight.detach().t().contiguous()
            b = linear.bias.detach().reshape(1, 1, -1)
            W_tt = self.ttnn.from_torch(W, device=device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
            b_tt = self.ttnn.from_torch(b, device=device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
            return W_tt, b_tt

        # Use module-level wrappers

        # Encoder heads
        self.enc_score_head_tt = TTNNLinearWrap(self.decoder_pt.enc_score_head, device, self.dtype, self.ttnn)
        self.enc_bbox_head_tt = TTNNMLPWrap(self.decoder_pt.enc_bbox_head, device, self.dtype, self.ttnn)
        # Decoder heads
        self.pre_bbox_head_tt = TTNNMLPWrap(self.decoder_pt.pre_bbox_head, device, self.dtype, self.ttnn)
        self.dec_score_head_tt = nn.ModuleList([
            TTNNLinearWrap(m, device, self.dtype, self.ttnn) for m in self.decoder_pt.dec_score_head
        ])
        self.dec_bbox_head_tt = nn.ModuleList([
            TTNNMLPWrap(m, device, self.dtype, self.ttnn) for m in self.decoder_pt.dec_bbox_head
        ])
        # Query pos head
        self.query_pos_head_tt = TTNNMLPWrap(self.decoder_pt.query_pos_head, device, self.dtype, self.ttnn)

    #Utils
    def _to_tt(self, x: torch.Tensor):
        return self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)

    def _to_torch(self, x_tt) -> torch.Tensor:
        return self.ttnn.to_torch(x_tt)

    def _enc_output_ttnn(self, memory_blc: torch.Tensor) -> torch.Tensor:
        """Apply TTNN Linear + LayerNorm to [B, L, C] memory."""
        ttnn = self.ttnn
        x_tt = self._to_tt(memory_blc)
        y = ttnn.linear(x_tt, self.W_enc, bias=self.b_enc)
        y = ttnn.layer_norm(y, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5)
        return self._to_torch(y)

    def _enc_score_ttnn(self, memory_blc: torch.Tensor) -> torch.Tensor:
        ttnn = self.ttnn
        x_tt = self._to_tt(memory_blc)
        y = ttnn.linear(x_tt, self.W_enc_score, bias=self.b_enc_score)
        return self._to_torch(y)

    def _topk_gather_ttnn(
        self,
        enc_logits: torch.Tensor,  # [B, L, C] or [B, L, 1]
        anchors: torch.Tensor,     # [B, L, 4]
        memory: torch.Tensor,      # [B, L, C_hidden]
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform on-device top-k over sequence dim and gather anchors/memory accordingly.

        Returns: (topk_scores[B,K], topk_indices[B,K], topk_anchors[B,K,4], topk_memory[B,K,C_hidden])
        """
        ttnn = self.ttnn
        B, L, _ = enc_logits.shape
        # Compute scores
        if enc_logits.shape[-1] == 1:
            scores = enc_logits.squeeze(-1)
            scores_tt = ttnn.from_torch(scores, device=self.device, dtype=self.dtype, layout=self.layout)
        else:
            logits_tt = ttnn.from_torch(enc_logits, device=self.device, dtype=self.dtype, layout=self.layout)
            # TTNN max returns values tensor when dim specified
            scores_tt = ttnn.max(logits_tt, dim=-1)

        # Top-k along sequence dim (1)
        topk_vals_tt, topk_idx_tt = ttnn.topk(scores_tt, k=k, dim=1, largest=True, sorted=True)
        topk_vals = ttnn.to_torch(topk_vals_tt)
        topk_idx = ttnn.to_torch(topk_idx_tt)

        # Prepare gather indices
        idx_anchor = topk_idx.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]).to(torch.int32)
        idx_memory = topk_idx.unsqueeze(-1).repeat(1, 1, memory.shape[-1]).to(torch.int32)

        anchors_tt = ttnn.from_torch(anchors, device=self.device, dtype=self.dtype, layout=self.layout)
        memory_tt = ttnn.from_torch(memory, device=self.device, dtype=self.dtype, layout=self.layout)

        idx_anchor_tt = ttnn.from_torch(idx_anchor, device=self.device, dtype=ttnn.uint32, layout=self.layout)
        idx_memory_tt = ttnn.from_torch(idx_memory, device=self.device, dtype=ttnn.uint32, layout=self.layout)

        topk_anchors_tt = ttnn.gather(anchors_tt, dim=1, index=idx_anchor_tt)
        topk_memory_tt = ttnn.gather(memory_tt, dim=1, index=idx_memory_tt)
        topk_anchors = ttnn.to_torch(topk_anchors_tt)
        topk_memory = ttnn.to_torch(topk_memory_tt)

        return topk_vals, topk_idx, topk_anchors, topk_memory

    #Full forward
    @torch.no_grad()
    def forward(self, feats: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        ttnn = self.ttnn
        # 1) Build encoder memory and shapes via PyTorch helper
        memory_blc, spatial_shapes = self.decoder_pt._get_encoder_input(list(feats))  # [B, L, C]

        # 2) enc_output + enc_score_head on device
        enc_out = self._enc_output_ttnn(memory_blc)
        # Use TTNN head for encoder score
        enc_logits = self.enc_score_head_tt(enc_out)

        # 3) Anchors + valid mask, and on-device top-k selection
        anchors, valid_mask = self.decoder_pt._generate_anchors(spatial_shapes, device=memory_blc.device)
        if memory_blc.shape[0] > 1:
            anchors = anchors.repeat(memory_blc.shape[0], 1, 1)
        memory_masked = valid_mask.to(memory_blc.dtype) * memory_blc

        topk_vals, topk_idx, topk_anchors, topk_memory = self._topk_gather_ttnn(
            enc_logits, anchors, memory_masked, self.decoder_pt.num_queries
        )
        # Ensure PyTorch heads receive float32 inputs
        topk_memory = topk_memory.float()
        topk_anchors = topk_anchors.float()

        # 4) Initial unactivated box offsets from encoder topk
        enc_topk_bbox_unact = self.enc_bbox_head_tt(topk_memory) + topk_anchors

        # 5) Prepare decoder initial states
        output = topk_memory.detach()
        ref_points_unact = enc_topk_bbox_unact.detach()
        ref_points_detach = torch.sigmoid(ref_points_unact)

        # Value list for MSDeformableAttention per-level
        # Mirror TransformerDecoder.value_op(None) path
        B, L, C = memory_blc.shape
        num_head = int(self.decoder_pt.nhead)
        head_dim = C // num_head
        split_shape = [h * w for h, w in spatial_shapes]
        value = memory_blc.reshape(B, L, num_head, -1).permute(0, 2, 3, 1).split(split_shape, dim=-1)

        # 6) Iterative decoding through layers
        eval_idx = int(self.decoder_pt.decoder.eval_idx if hasattr(self.decoder_pt.decoder, 'eval_idx') else len(self.layers) - 1)
        _ = getattr(self.decoder_pt.decoder, 'lqe_layers', None)  # placeholder for future TTNN port

        up = float(self.decoder_pt.up)
        reg_scale = float(self.decoder_pt.reg_scale)
        project = weighting_function(int(self.decoder_pt.reg_max), torch.tensor([up]), reg_scale, deploy=True)

        dec_out_bboxes: List[torch.Tensor] = []
        dec_out_logits: List[torch.Tensor] = []

        output_detach: Optional[torch.Tensor] = None
        prev_pred_corners: Optional[torch.Tensor] = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            # Use TTNN query_pos_head (omit clamp for now)
            query_pos_embed = self.query_pos_head_tt(ref_points_detach)

            # Execute TTNN decoder layer
            output = layer(
                output, ref_points_input, list(value), spatial_shapes, attn_mask=None, query_pos_embed=query_pos_embed
            )
            # Ensure FP32 when feeding any PyTorch heads next
            output = output.float()

            if i == 0:
                pre_bboxes = torch.sigmoid(self.pre_bbox_head_tt(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = self.dec_score_head_tt[0](output)
                ref_points_initial = pre_bboxes.detach()

            residual = output_detach if output_detach is not None else 0.0
            carry = prev_pred_corners if prev_pred_corners is not None else 0.0
            pred_corners = self.dec_bbox_head_tt[i](output + residual) + carry
            inter_ref_bbox = distance2bbox(ref_points_initial, self.integral(pred_corners, project), reg_scale)

            if i == eval_idx:
                scores = self.dec_score_head_tt[i](output)
                # Skip optional lqe_layers for now (TTNN port pending)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            prev_pred_corners = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        out = {"pred_logits": dec_out_logits[-1], "pred_boxes": dec_out_bboxes[-1]}
        return out

    #print debug
    @torch.no_grad()
    def forward_debug(self, feats: Sequence[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """Compute initial encoder memory and CPU top-k selection.

        Returns:
          out: dict with keys 'enc_topk_logits' and 'enc_topk_bbox_unact' (CPU torch tensors)
          debug: stage artifacts including 'memory', 'enc_output', 'enc_logits' and selection tensors
        """
        debug: Dict[str, object] = {}

        # Build memory via PyTorch projection + flattening
        memory_blc, spatial_shapes = self.decoder_pt._get_encoder_input(list(feats))
        debug["memory"] = memory_blc
        debug["spatial_shapes"] = spatial_shapes

        # Apply enc_output (TTNN) and enc_score_head (TTNN)
        enc_out = self._enc_output_ttnn(memory_blc)
        debug["enc_output"] = enc_out
        enc_logits = self._enc_score_ttnn(enc_out)
        debug["enc_logits"] = enc_logits

        # Generate anchors (CPU) and valid mask
        anchors, valid_mask = self.decoder_pt._generate_anchors(spatial_shapes, device=memory_blc.device)
        if memory_blc.shape[0] > 1:
            anchors = anchors.repeat(memory_blc.shape[0], 1, 1)
        memory_masked = valid_mask.to(memory_blc.dtype) * memory_blc

        # CPU top-k selection
        if enc_logits.shape[-1] == 1:
            scores = enc_logits.squeeze(-1)
        else:
            scores = enc_logits.max(dim=-1).values
        topk_scores, topk_ind = torch.topk(scores, self.decoder_pt.num_queries, dim=1)
        topk_anchors = anchors.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
        topk_memory = memory_masked.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, memory_masked.shape[-1]))

        # CPU bbox head to validate first step end-to-end (unact)
        enc_topk_bbox_unact = self.decoder_pt.enc_bbox_head(topk_memory) + topk_anchors

        out = {
            "enc_topk_logits": topk_scores if enc_logits.shape[-1] == 1 else enc_logits.gather(
                1, topk_ind.unsqueeze(-1)
            ).squeeze(-1),
            "enc_topk_bbox_unact": enc_topk_bbox_unact,
        }

        debug["topk_ind"] = topk_ind
        debug["topk_scores"] = topk_scores
        debug["topk_anchors"] = topk_anchors
        debug["topk_memory"] = topk_memory

        return out, debug

    def close(self):
        if self._owns_device:
            try:
                self.ttnn.close_device(self.device)
            except Exception:
                pass


class TTNNMSDeformableAttention(nn.Module):
    """TTNN port of MSDeformableAttention using ttnn.grid_sample.

    This module consumes the parameters from a PyTorch MSDeformableAttention instance
    and reproduces its forward pass using TTNN primitives where appropriate.
    """

    def __init__(self, msda_pt: nn.Module, device, dtype=None, layout=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT

        self.embed_dim = int(msda_pt.embed_dim)
        self.num_heads = int(msda_pt.num_heads)
        self.num_levels = int(msda_pt.num_levels)
        self.num_points_list: List[int] = list(msda_pt.num_points_list)
        self.total_points = int(msda_pt.total_points)
        self.method = msda_pt.method
        self.offset_scale = float(getattr(msda_pt, 'offset_scale', 0.5))

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        # Port sampling_offsets, attention_weights linears
        self.W_off = ttnn.from_torch(
            msda_pt.sampling_offsets.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.b_off = ttnn.from_torch(
            msda_pt.sampling_offsets.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.W_attn = ttnn.from_torch(
            msda_pt.attention_weights.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.b_attn = ttnn.from_torch(
            msda_pt.attention_weights.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )

    def _to_tt(self, x: torch.Tensor):
        return self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)

    def forward(
        self,
        query: torch.Tensor,                     # [B, Lq, C]
        reference_points: torch.Tensor,          # [B, Lq, num_levels, 2 or 4]
        value: List[torch.Tensor],               # list of [B, H, C_head, H_lv, W_lv] with H=num_heads
        value_spatial_shapes: List[List[int]],   # [[H1,W1], [H2,W2], ...]
    ) -> torch.Tensor:
        ttnn = self.ttnn
        B, Lq, C = query.shape

        # Offsets and attention weights via TTNN linear
        q_tt = self._to_tt(query)
        off = ttnn.linear(q_tt, self.W_off, bias=self.b_off)  # [B,Lq,total_points*2]
        attn = ttnn.linear(q_tt, self.W_attn, bias=self.b_attn)  # [B,Lq,total_points]
        sampling_offsets = ttnn.to_torch(off).reshape(B, Lq, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = torch.softmax(ttnn.to_torch(attn).reshape(B, Lq, self.num_heads, sum(self.num_points_list)), dim=-1)

        # Compute sampling locations in [0,1] per level
        if reference_points.shape[-1] == 2:
            # offset_normalizer [1,1,1,L,1,2] with (w,h)
            offset_normalizer = torch.tensor(value_spatial_shapes, dtype=query.dtype, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations_per_level = []
            off_splits = sampling_offsets.split(self.num_points_list, dim=3)
            for lvl, (h, w) in enumerate(value_spatial_shapes):
                loc = reference_points[:, :, None, lvl, None, :] + off_splits[lvl] / offset_normalizer[:, :, :, lvl, :, :]
                # Result [B,Lq,H,num_points_lvl,2]
                sampling_locations_per_level.append(loc)
        elif reference_points.shape[-1] == 4:
            # Some pipelines provide a single level reference and expect broadcasting to all levels
            if reference_points.shape[2] == 1 and self.num_levels > 1:
                reference_points = reference_points.repeat(1, 1, self.num_levels, 1)
            # Compute per-level offsets without relying on ambiguous broadcasting
            off_splits = sampling_offsets.split(self.num_points_list, dim=3)
            sampling_locations_per_level = []
            for lvl in range(self.num_levels):
                npts = float(self.num_points_list[lvl])
                # Scale offsets per point count and level-specific (w,h)
                off_lvl = (
                    off_splits[lvl] * (1.0 / npts) * reference_points[:, :, None, lvl, None, 2:] * self.offset_scale
                )
                loc = reference_points[:, :, None, lvl, None, :2] + off_lvl
                sampling_locations_per_level.append(loc)
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        # Prepare attention_weights per level
        attn_splits = attention_weights.split(self.num_points_list, dim=3)

        # Accumulate per-level sampled values
        sampled_sum = None  # [B*H, C_head, Lq]
        for lvl, (h, w) in enumerate(value_spatial_shapes):
            # value[lvl]: [B, H, C_head, H_lv*W_lv] or [B,H,C_head,H_lv,W_lv]
            v = value[lvl]
            if v.dim() == 4:
                v = v.reshape(B, self.num_heads, self.head_dim, h, w)
            else:
                assert v.dim() == 5
            Bh = B * self.num_heads
            # NHWC input for TTNN grid_sample
            v_nhwc = v.permute(0, 1, 3, 4, 2).reshape(Bh, h, w, self.head_dim).contiguous()
            c_pad = (32 - (self.head_dim % 32)) % 32
            if c_pad != 0:
                v_nhwc = torch.nn.functional.pad(v_nhwc, (0, c_pad))
            v_tt = ttnn.from_torch(v_nhwc, device=self.device, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

            # Grid for this level: [B, Lq, H, num_points_lvl, 2] -> reshape to [Bh, Lq, num_points_lvl, 2]
            loc = sampling_locations_per_level[lvl]  # [B,Lq,H,num_pts,2]
            # Normalize to [-1, 1]
            grid = loc.clone()
            grid[..., 0] = grid[..., 0] * 2.0 - 1.0  # x normalized
            grid[..., 1] = grid[..., 1] * 2.0 - 1.0  # y normalized
            grid = grid.permute(0, 2, 1, 3, 4).reshape(Bh, Lq, self.num_points_list[lvl], 2).contiguous()
            grid_tt = ttnn.from_torch(grid.to(torch.bfloat16), device=self.device)

            # Sample: output (Bh, Lq, P, C)
            out = ttnn.grid_sample(v_tt, grid_tt, mode="bilinear", padding_mode="zeros", use_precomputed_grid=False)
            out_t = ttnn.to_torch(out)  # (Bh, Lq, P, C_pad)
            if c_pad != 0:
                out_t = out_t[..., : self.head_dim]
            out_chw = out_t.permute(0, 3, 1, 2).contiguous()  # (Bh, C, Lq, P)

            # Apply attention weights for this level: [B, Lq, H, P] -> [Bh, 1, Lq, P]
            aw = attn_splits[lvl]  # [B,Lq,H,P]
            aw_bh = aw.permute(0, 2, 1, 3).reshape(Bh, 1, Lq, self.num_points_list[lvl])
            weighted = out_chw * aw_bh
            # Sum over points
            wsum = weighted.sum(dim=3)  # (Bh, C, Lq)

            sampled_sum = wsum if sampled_sum is None else sampled_sum + wsum

        # Reshape back: (Bh, C, Lq) -> (B, H, C_head, Lq)
        out = sampled_sum.reshape(B, self.num_heads, self.head_dim, Lq).permute(0, 3, 1, 2).reshape(B, Lq, self.embed_dim)
        return out


class TTNNTransformerDecoderLayer(nn.Module):
    def __init__(self, layer_pt: nn.Module, device, dtype=None, layout=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT

        self.d_model = int(layer_pt.self_attn.embed_dim)
        self.n_head = int(layer_pt.self_attn.num_heads)

        # Self-attention
        from .mha_ttnn import TTNNMHA
        self.self_attn = TTNNMHA(layer_pt.self_attn, device, dtype=self.dtype, layout=self.layout)
        # Norms
        self.ln1_w = ttnn.from_torch(layer_pt.norm1.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln1_b = ttnn.from_torch(layer_pt.norm1.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln3_w = ttnn.from_torch(layer_pt.norm3.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln3_b = ttnn.from_torch(layer_pt.norm3.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        # Cross-attention via MSDeformableAttention
        self.cross_attn = TTNNMSDeformableAttention(layer_pt.cross_attn, device, dtype=self.dtype, layout=self.layout)

        # Gate
        # gate is linear on concat(target, cross): 2*d_model -> 2*d_model
        self.W_gate = ttnn.from_torch(layer_pt.gateway.gate.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.b_gate = ttnn.from_torch(layer_pt.gateway.gate.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln_g_w = ttnn.from_torch(layer_pt.gateway.norm.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln_g_b = ttnn.from_torch(layer_pt.gateway.norm.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        # FFN
        self.W1 = ttnn.from_torch(layer_pt.linear1.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.b1 = ttnn.from_torch(layer_pt.linear1.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.W2 = ttnn.from_torch(layer_pt.linear2.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.b2 = ttnn.from_torch(layer_pt.linear2.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        # Activation
        self._act = ttnn.gelu if isinstance(layer_pt.activation, nn.GELU) else ttnn.relu

    def _ln(self, x_tt, w, b):
        return self.ttnn.layer_norm(x_tt, weight=w, bias=b, epsilon=1e-5)

    def forward(self, target: torch.Tensor, reference_points: torch.Tensor, value_list: List[torch.Tensor], spatial_shapes: List[List[int]], attn_mask=None, query_pos_embed=None) -> torch.Tensor:
        # Self-attn
        q = target if query_pos_embed is None else (target + query_pos_embed)
        sa = self.self_attn(q, x_k=q, x_v=target)
        x = target + sa
        x_tt = self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
        x_tt = self._ln(x_tt, self.ln1_w, self.ln1_b)
        x = self.ttnn.to_torch(x_tt)

        # Cross-attn
        qpos = x if query_pos_embed is None else (x + query_pos_embed)
        ca = self.cross_attn(qpos, reference_points, value_list, spatial_shapes)

        # Gate
        gate_in = torch.cat([x, ca], dim=-1)
        gi_tt = self.ttnn.from_torch(gate_in, device=self.device, dtype=self.dtype, layout=self.layout)
        gates = self.ttnn.linear(gi_tt, self.W_gate, bias=self.b_gate)
        gates_t = self.ttnn.to_torch(gates)
        g1, g2 = torch.sigmoid(gates_t).chunk(2, dim=-1)
        gx = g1 * x + g2 * ca
        gx_tt = self.ttnn.from_torch(gx, device=self.device, dtype=self.dtype, layout=self.layout)
        gx_tt = self._ln(gx_tt, self.ln_g_w, self.ln_g_b)
        x = self.ttnn.to_torch(gx_tt)

        # FFN
        x_tt = self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
        y = self.ttnn.linear(x_tt, self.W1, bias=self.b1)
        y = self._act(y)
        y = self.ttnn.linear(y, self.W2, bias=self.b2)
        y_t = self.ttnn.to_torch(y)
        x = x + y_t
        x_tt = self.ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
        x_tt = self._ln(x_tt, self.ln3_w, self.ln3_b)
        return self.ttnn.to_torch(x_tt)


class TTNNIntegral(nn.Module):
    def __init__(self, reg_max: int, device, dtype=None, layout=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self.reg_max = int(reg_max)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4*(reg_max+1)]
        ttnn = self.ttnn
        B, L, D = x.shape
        R = self.reg_max + 1
        x_flat = x.reshape(-1, R)
        # Compute softmax in higher precision on host to reduce drift
        probs_t = torch.softmax(x_flat.float(), dim=-1)
        probs = ttnn.from_torch(probs_t.to(dtype=torch.bfloat16), device=self.device, dtype=self.dtype, layout=self.layout)
        proj_tt = ttnn.from_torch(project.reshape(-1, 1), device=self.device, dtype=self.dtype, layout=self.layout)
        y = ttnn.matmul(probs, proj_tt)
        y_t = ttnn.to_torch(y).reshape(B, L, -1)
        return y_t
