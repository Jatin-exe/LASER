from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import ttnn


class TTNNMHA(nn.Module):
    """TTNN-backed Multi-Head Attention using SDPA + linear layers.

    Mirrors torch.nn.MultiheadAttention (batch_first=True, no dropout) for inference.
    """

    def __init__(self, mha_pt: nn.MultiheadAttention, device, dtype=None, layout=None):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.embed_dim = int(mha_pt.embed_dim)
        self.num_heads = int(mha_pt.num_heads)
        assert getattr(mha_pt, "batch_first", False), "Expected batch_first=True"
        assert (self.embed_dim % self.num_heads) == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT

        # Extract QKV and out_proj
        W_qkv = mha_pt.in_proj_weight.detach().clone()  # [3*E, E]
        b_qkv = mha_pt.in_proj_bias.detach().clone()    # [3*E]
        self.W_q, self.W_k, self.W_v = torch.chunk(W_qkv, 3, dim=0)
        self.b_q, self.b_k, self.b_v = torch.chunk(b_qkv, 3, dim=0)

        self.W_o = mha_pt.out_proj.weight.detach().clone()  # [E, E]
        self.b_o = mha_pt.out_proj.bias.detach().clone()    # [E]

        # Materialize on device
        def to_tt(t):
            return ttnn.from_torch(t, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        self.W_q_tt = to_tt(self.W_q.t().contiguous())
        self.W_k_tt = to_tt(self.W_k.t().contiguous())
        self.W_v_tt = to_tt(self.W_v.t().contiguous())
        self.b_q_tt = to_tt(self.b_q.reshape(1, 1, -1))
        self.b_k_tt = to_tt(self.b_k.reshape(1, 1, -1))
        self.b_v_tt = to_tt(self.b_v.reshape(1, 1, -1))

        self.W_o_tt = to_tt(self.W_o.t().contiguous())
        self.b_o_tt = to_tt(self.b_o.reshape(1, 1, -1))

        # Choose SDPA fast-path only when head_dim >= 32 to avoid unsupported padding
        self._use_sdpa = (self.head_dim >= 32)

        # Optimized fused-QKV path (optional). Use when transformer helpers are available.
        self._use_fused_qkv = False
        try:
            # Build fused QKV weights for a single linear op: [3E, E]
            W_qkv_fused = torch.cat([self.W_q, self.W_k, self.W_v], dim=0)
            b_qkv_fused = torch.cat([self.b_q, self.b_k, self.b_v], dim=0)
            self.W_qkv_tt = to_tt(W_qkv_fused.t().contiguous())  # [E,3E]
            # Put bias in L1 for faster access if available
            self.b_qkv_tt = ttnn.from_torch(
                b_qkv_fused.reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
            )
            # Verify transformer helpers exist
            _ = getattr(ttnn, "transformer")
            # Enable fused path by default
            self._use_fused_qkv = True
        except Exception:
            self._use_fused_qkv = False

    def _linear(self, x_tt, W_tt, b_tt):
        # x_tt: [B, S, E], W_tt: [E_out, E_in] matching torch linear semantics
        ttnn = self.ttnn
        y = ttnn.linear(x_tt, W_tt, bias=b_tt)
        return y

    def _reshape_to_qkv(self, x_tt, b: int, s: int):
        # x_tt: [B, S, E] => [B, H, S, Dh]
        ttnn = self.ttnn
        y = ttnn.reshape(x_tt, (b, s, self.num_heads, self.head_dim))
        y = ttnn.permute(y, (0, 2, 1, 3))
        return y

    def _merge_heads(self, x_tt, b: int, s: int):
        # x_tt: [B, H, S, Dh] => [B, S, E]
        ttnn = self.ttnn
        y = ttnn.permute(x_tt, (0, 2, 1, 3))
        y = ttnn.reshape(y, (b, s, self.num_heads * self.head_dim))
        return y

    def forward(self, x_q: torch.Tensor, x_k: Optional[torch.Tensor] = None, x_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        ttnn = self.ttnn
        # Convert to device tensor [B, S, E]
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        b, s, _ = x_q.shape
        xq_tt = ttnn.from_torch(x_q, device=self.device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
        if self._use_fused_qkv and (x_k is x_q) and (x_v is x_q):
            # Fused QKV single linear
            fused = ttnn.linear(xq_tt, self.W_qkv_tt, bias=self.b_qkv_tt)
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                fused, num_heads=self.num_heads
            )
            # q,k,v now in [B,H,S,Dh] layouts suitable for attention ops
        else:
            xk_tt = xq_tt if x_k is x_q else ttnn.from_torch(x_k, device=self.device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
            xv_tt = xq_tt if x_v is x_q else ttnn.from_torch(x_v, device=self.device, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)
            # Separate projections
            q = self._linear(xq_tt, self.W_q_tt, self.b_q_tt)
            k = self._linear(xk_tt, self.W_k_tt, self.b_k_tt)
            v = self._linear(xv_tt, self.W_v_tt, self.b_v_tt)
            # Reshape to [B, H, S, Dh]
            q = self._reshape_to_qkv(q, b, s)
            k = self._reshape_to_qkv(k, b, s)
            v = self._reshape_to_qkv(v, b, s)

        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        if self._use_sdpa:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale
            )
        else:
            # Manual attention via BMM to support small head_dim
            # Flatten heads into batch: [B*H, S, Dh]
            bh = ttnn.reshape(q, (b * self.num_heads, s, self.head_dim))
            bk = ttnn.reshape(k, (b * self.num_heads, s, self.head_dim))
            bv = ttnn.reshape(v, (b * self.num_heads, s, self.head_dim))
            # q @ k^T -> [B*H, S, S]
            bkt = ttnn.permute(bk, (0, 2, 1))
            scores = ttnn.matmul(bh, bkt)
            scores = ttnn.multiply(scores, scale)
            probs = ttnn.softmax(scores)
            ctx = ttnn.matmul(probs, bv)
            # Back to [B, H, S, Dh]
            attn_out = ttnn.reshape(ctx, (b, self.num_heads, s, self.head_dim))

        # Merge heads and out_proj
        y = self._merge_heads(attn_out, b, s)
        y = self._linear(y, self.W_o_tt, self.b_o_tt)

        y_torch = ttnn.to_torch(y)
        return y_torch
