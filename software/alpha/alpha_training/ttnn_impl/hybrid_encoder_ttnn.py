"""TTNN mirror of the HybridEncoder used by DFINE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import ttnn


from .mha_ttnn import TTNNMHA

# helpers and wrappers
from .hgnetv2_ttnn_manual import (
    TTActivation,
    TTNNConv2d,
    _torch_to_tt_activation,
    _tt_activation_to_torch,
    _activation_to_nhwc,
    _activation_from_nhwc,
    fold_bn_to_conv,
)


@dataclass
class _TTNNConfig:
    dtype: object
    layout: object


class _TTNNConvBN(nn.Module):
    """Conv2d with BN folded into weights/bias using TTNNConv2d"""

    def __init__(self, device, conv_pt: nn.Conv2d, bn_pt: nn.BatchNorm2d, cfg: _TTNNConfig):
        super().__init__()
        weight_t, bias_t = fold_bn_to_conv(conv_pt, bn_pt)
        padding = tuple(int(p) for p in conv_pt.padding)
        self.conv = TTNNConv2d(
            device,
            weight_t,
            bias_t,
            stride=tuple(conv_pt.stride),
            padding=padding,
            dilation=tuple(conv_pt.dilation),
            groups=int(conv_pt.groups),
            dtype=cfg.dtype,
            layout=None,  # use default ROW_MAJOR output layout from TTNNConv2d
        )

    def forward(self, act: TTActivation) -> TTActivation:
        return self.conv(act)

    def enable_l1(self):
        if hasattr(self.conv, "use_l1_output"):
            self.conv.use_l1_output()


class _TTNNTransformerEncoderLayer(nn.Module):
    def __init__(self, layer_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        self.ttnn = ttnn
        self.device = device
        self.dtype = cfg.dtype
        # Force TILE layout for Transformer internals (layernorm/linear/matmul)
        # perf diff btw TILE and Row major ? 
        self.layout = ttnn.TILE_LAYOUT
        self.normalize_before = getattr(layer_pt, "normalize_before", False)

        # MHA
        self.mha = TTNNMHA(layer_pt.self_attn, device, dtype=self.dtype, layout=self.layout)

        # Feed-forward
        self.W1 = ttnn.from_torch(
            layer_pt.linear1.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.b1 = ttnn.from_torch(
            layer_pt.linear1.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.W2 = ttnn.from_torch(
            layer_pt.linear2.weight.detach().t().contiguous(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.b2 = ttnn.from_torch(
            layer_pt.linear2.bias.detach().reshape(1, 1, -1), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
        )

        # Norms
        self.ln1_weight = ttnn.from_torch(layer_pt.norm1.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln1_bias = ttnn.from_torch(layer_pt.norm1.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln2_weight = ttnn.from_torch(layer_pt.norm2.weight.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.ln2_bias = ttnn.from_torch(layer_pt.norm2.bias.detach(), device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        # Activation
        if isinstance(layer_pt.activation, nn.GELU):
            self._act = ttnn.gelu
        else:
            self._act = ttnn.relu

    def _ln(self, x_tt, weight, bias):
        return self.ttnn.layer_norm(x_tt, weight=weight, bias=bias, epsilon=1e-5)

    def _linear(self, x_tt, W, b):
        return self.ttnn.linear(x_tt, W, bias=b)

    def forward(self, src_flatten: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src_flatten: [B, S, C] torch tensor
        ttnn = self.ttnn
        B, S, C = src_flatten.shape
        # residual 1
        x = src_flatten
        if self.normalize_before:
            x_tt = ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
            x_tt = self._ln(x_tt, self.ln1_weight, self.ln1_bias)
            x = ttnn.to_torch(x_tt)
        qk = x if pos_embed is None else (x + pos_embed)
        attn = self.mha(qk, x_k=qk, x_v=x)
        x = x + attn
        if not self.normalize_before:
            x_tt = ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
            x_tt = self._ln(x_tt, self.ln1_weight, self.ln1_bias)
            x = ttnn.to_torch(x_tt)

        # residual 2
        if self.normalize_before:
            x_tt = ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
            x_tt = self._ln(x_tt, self.ln2_weight, self.ln2_bias)
            x = ttnn.to_torch(x_tt)
        x_tt = ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
        x_tt = self._linear(x_tt, self.W1, self.b1)
        x_tt = self._act(x_tt)
        x_tt = self._linear(x_tt, self.W2, self.b2)
        x = x + ttnn.to_torch(x_tt)
        if not self.normalize_before:
            x_tt = ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=self.layout)
            x_tt = self._ln(x_tt, self.ln2_weight, self.ln2_bias)
            x = ttnn.to_torch(x_tt)
        return x


# FPN/PAN section
def _act_op_from_module(act_mod):
    if act_mod is None:
        return None
    name = act_mod.__class__.__name__
    if name.lower().startswith("identity"):
        return None
    if name.lower().startswith("silu") or name.lower().startswith("swish"):
        return getattr(ttnn, "silu", getattr(ttnn, "hardswish", ttnn.gelu))
    if name.lower().startswith("gelu"):
        return ttnn.gelu
    if name.lower().startswith("relu6"):
        return getattr(ttnn, "relu6", ttnn.relu)
    if name.lower().startswith("relu"):
        return ttnn.relu
    if name.lower().startswith("leakyrelu"):
        return getattr(ttnn, "leaky_relu", ttnn.relu)
    if name.lower().startswith("hardsigmoid"):
        return getattr(ttnn, "hardsigmoid", ttnn.sigmoid)
    return None


class _TTNNConvBNActTorch(nn.Module):
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        conv_pt = getattr(module_pt, "conv")
        bn_pt = getattr(module_pt, "norm")
        self.conv = _TTNNConvBN(device, conv_pt, bn_pt, cfg)
        self.act_op = _act_op_from_module(getattr(module_pt, "act", None))
        self.device = device
        self.cfg = cfg
        # Prefer L1 outputs for these small convs
        try:
            self.conv.enable_l1()
        except Exception:
            pass

    def _apply_act(self, y: TTActivation) -> TTActivation:
        if self.act_op is None:
            return y
        tens = ttnn.to_layout(y.tensor, ttnn.TILE_LAYOUT)
        z = self.act_op(tens)
        return TTActivation(z, y.batch, y.height, y.width, y.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = _torch_to_tt_activation(x, self.device, self.cfg.dtype, self.cfg.layout)
        y = self.conv(act)
        y = self._apply_act(y)
        return _tt_activation_to_torch(y)


class _TTNNVGGBlockTorch(nn.Module):
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        # conv1: ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # conv2: ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.conv1 = _TTNNConvBNActTorch(module_pt.conv1, device, cfg)
        self.conv2 = _TTNNConvBNActTorch(module_pt.conv2, device, cfg)
        self.act_op = _act_op_from_module(getattr(module_pt, "act", None))
        self.device = device
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = y1 + y2
        if self.act_op is not None:
            act = _torch_to_tt_activation(y, self.device, self.cfg.dtype, self.cfg.layout)
            tens = ttnn.to_layout(act.tensor, ttnn.TILE_LAYOUT)
            z = self.act_op(tens)
            y = _tt_activation_to_torch(TTActivation(z, act.batch, act.height, act.width, act.channels))
        return y


class _TTNNCSPLayerTorch(nn.Module):
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        self.conv1 = _TTNNConvBNActTorch(module_pt.conv1, device, cfg)
        self.conv2 = _TTNNConvBNActTorch(module_pt.conv2, device, cfg)
        self.bottlenecks = nn.ModuleList([
            _TTNNVGGBlockTorch(b, device, cfg) for b in module_pt.bottlenecks
        ])
        self.has_conv3 = not isinstance(module_pt.conv3, nn.Identity)
        self.conv3 = (
            _TTNNConvBNActTorch(module_pt.conv3, device, cfg) if self.has_conv3 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        for b in self.bottlenecks:
            x1 = b(x1)
        x2 = self.conv2(x)
        out = x1 + x2
        out = self.conv3(out) if self.has_conv3 else out
        return out


class _TTNNRepNCSPELAN4Torch(nn.Module):
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        self.cv1 = _TTNNConvBNActTorch(module_pt.cv1, device, cfg)
        self.cv2_csp = _TTNNCSPLayerTorch(module_pt.cv2[0], device, cfg)
        self.cv2_post = _TTNNConvBNActTorch(module_pt.cv2[1], device, cfg)
        self.cv3_csp = _TTNNCSPLayerTorch(module_pt.cv3[0], device, cfg)
        self.cv3_post = _TTNNConvBNActTorch(module_pt.cv3[1], device, cfg)
        self.cv4 = _TTNNConvBNActTorch(module_pt.cv4, device, cfg)
        self.c = module_pt.c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b = y.split((self.c, self.c), dim=1)
        y2 = self.cv2_post(self.cv2_csp(b))
        y3 = self.cv3_post(self.cv3_csp(y2))
        y_cat = torch.concat([a, b, y2, y3], dim=1)
        out = self.cv4(y_cat)
        return out


class _TTNNSCDownTorch(nn.Module):
    def __init__(self, module_pt: nn.Module, device, cfg: _TTNNConfig):
        super().__init__()
        self.cv1 = _TTNNConvBNActTorch(module_pt.cv1, device, cfg)
        self.cv2 = _TTNNConvBNActTorch(module_pt.cv2, device, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class HybridEncoderTTNN(nn.Module):
    """TTNN port of the HybridEncoder used in DFINE."""

    def __init__(self, encoder_pt: nn.Module, device=None, device_id: int = 0, return_stage: str = "proj"):
        super().__init__()
        self.ttnn = ttnn

        # Open a device if a shared one is not provided (prefer sharing backbone device)
        self._owns_device = device is None
        if device is None:
            try:
                device = ttnn.open_device(device_id=device_id, l1_small_size=655360)
            except TypeError:
                device = ttnn.open_device(device_id=device_id)
        self.device = device
        # Controls what forward() returns: "proj" for just input projections (unit tests),
        # "final" for full FPN+PAN outputs (deployment/benchmarks).
        assert return_stage in ("proj", "final"), "return_stage must be 'proj' or 'final'"
        self._return_stage = return_stage

        # Prefer ROW_MAJOR layout for encoder to minimize reshape/view restrictions across ops
        self.cfg = _TTNNConfig(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Mirror key attrs from PyTorch encoder for consistent behavior later
        self.in_channels: List[int] = list(getattr(encoder_pt, "in_channels"))
        self.feat_strides: List[int] = list(getattr(encoder_pt, "feat_strides"))
        self.hidden_dim: int = int(getattr(encoder_pt, "hidden_dim"))
        self.use_encoder_idx: List[int] = list(getattr(encoder_pt, "use_encoder_idx"))
        self.num_encoder_layers: int = int(getattr(encoder_pt, "num_encoder_layers"))
        self.pe_temperature: float = float(getattr(encoder_pt, "pe_temperature"))
        self.eval_spatial_size = getattr(encoder_pt, "eval_spatial_size")

        # 2.1 Input projections: build TTNN convs from PyTorch conv+bn
        self.input_proj = nn.ModuleList()
        for proj_pt in encoder_pt.input_proj:
            # Each proj is an OrderedDict inside nn.Sequential with named modules
            conv_pt = getattr(proj_pt, "conv")
            bn_pt = getattr(proj_pt, "norm")
            self.input_proj.append(_TTNNConvBN(self.device, conv_pt, bn_pt, self.cfg))

        # Keep reference to PyTorch encoder for complex blocks (temporary during bring-up)
        self._encoder_pt = encoder_pt

        # Lateral convs (1x1) for FPN
        self.lateral_convs_tt = nn.ModuleList()
        for lat in encoder_pt.lateral_convs:
            conv_pt = getattr(lat, "conv")
            bn_pt = getattr(lat, "norm")
            self.lateral_convs_tt.append(_TTNNConvBN(self.device, conv_pt, bn_pt, self.cfg))

        # FPN fusion blocks (RepNCSPELAN4)
        self.fpn_blocks_tt = nn.ModuleList(
            [_TTNNRepNCSPELAN4Torch(m, self.device, self.cfg) for m in encoder_pt.fpn_blocks]
        )

        # PAN blocks
        self.downsample_convs_tt = nn.ModuleList(
            [_TTNNSCDownTorch(seq[0], self.device, self.cfg) for seq in encoder_pt.downsample_convs]
        )
        self.pan_blocks_tt = nn.ModuleList(
            [_TTNNRepNCSPELAN4Torch(m, self.device, self.cfg) for m in encoder_pt.pan_blocks]
        )

        # 2.2 Encoder layers (optional)
        self._encoder_layers: Optional[nn.ModuleList] = None

        if self.num_encoder_layers > 0:
            # Build TTNN encoder stacks matching selected indices
            self._encoder_layers = nn.ModuleList()
            for i, enc_ind in enumerate(self.use_encoder_idx):
                enc_stack = nn.ModuleList()
                # encoder_pt.encoder[i] is a TransformerEncoder with layers list
                enc_block_pt = encoder_pt.encoder[i]
                for lyr_pt in enc_block_pt.layers:
                    enc_stack.append(_TTNNTransformerEncoderLayer(lyr_pt, self.device, self.cfg))
                self._encoder_layers.append(enc_stack)


    def _to_ttnn(self, x: torch.Tensor) -> TTActivation:
        ttnn_mod = self.ttnn
        n, c, h, w = x.shape
        nhwc = x.permute(0, 2, 3, 1).contiguous()
        nhwc_tt = ttnn_mod.from_torch(
            nhwc, device=self.device, dtype=self.cfg.dtype, layout=ttnn_mod.ROW_MAJOR_LAYOUT
        )
        flat = ttnn_mod.reshape(nhwc_tt, (1, 1, n * h * w, c))
        return TTActivation(flat, n, h, w, c)

    def _to_torch(self, act: TTActivation) -> torch.Tensor:
        return _tt_activation_to_torch(act)

    def _upsample2x_ttnn(self, x: torch.Tensor) -> torch.Tensor:
        ttnn_mod = self.ttnn
        b, c, h, w = x.shape
        act = self._to_ttnn(x)
        nhwc = _activation_to_nhwc(act)
        up = ttnn_mod.repeat(nhwc, (1, 2, 2, 1))
        up_act = _activation_from_nhwc(up, b, h * 2, w * 2, c)
        return self._to_torch(up_act)

    def _project_inputs(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        projected: List[torch.Tensor] = []
        for idx, feat in enumerate(feats):
            act = self._to_ttnn(feat)
            proj_act = self.input_proj[idx](act)
            projected.append(self._to_torch(proj_act))
        return projected

    def _run_encoder_layers(self, proj_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self._encoder_layers:
            return proj_feats
        encoded = list(proj_feats)
        for stack_id, enc_ind in enumerate(self.use_encoder_idx):
            x_feat = proj_feats[enc_ind]
            b, c, h, w = x_feat.shape
            src_flatten = x_feat.flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)
            else:
                cached = getattr(self, f"pos_embed{enc_ind}", None)
                if cached is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = cached.to(src_flatten.device)
            for layer in self._encoder_layers[stack_id]:
                src_flatten = layer(src_flatten, pos_embed=pos_embed)
            encoded[enc_ind] = src_flatten.permute(0, 2, 1).reshape(b, self.hidden_dim, h, w).contiguous()
        return encoded

    def _run_fpn(self, encoded_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        inner: List[torch.Tensor] = [encoded_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            hi = inner[0]
            lo = encoded_feats[idx - 1]
            lateral = self.lateral_convs_tt[len(self.in_channels) - 1 - idx]

            hi = self._to_torch(lateral(self._to_ttnn(hi)))

            inner[0] = hi
            upsampled = self._upsample2x_ttnn(hi)
            fused = torch.cat([upsampled, lo], dim=1)
            block = self.fpn_blocks_tt[len(self.in_channels) - 1 - idx]
            inner.insert(0, block(fused))
        return inner

    def _run_pan(self, fpn_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = [fpn_feats[0]]
        for idx in range(len(self.in_channels) - 1):
            low = outs[-1]
            high = fpn_feats[idx + 1]
            down = self.downsample_convs_tt[idx](low)
            fused = torch.cat([down, high], dim=1)
            outs.append(self.pan_blocks_tt[idx](fused))
        return outs

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        proj_feats = self._project_inputs(feats)
        if self._return_stage == "proj":
            return proj_feats
        encoded = self._run_encoder_layers(proj_feats)
        fpn_feats = self._run_fpn(encoded)
        return self._run_pan(fpn_feats)

    def forward_debug(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        debug: Dict[str, object] = {}
        proj_feats = self._project_inputs(feats)
        debug["proj"] = proj_feats
        if self._return_stage == "proj":
            debug["encoder"] = proj_feats
            debug["final"] = proj_feats
            return proj_feats, debug

        encoded = self._run_encoder_layers(proj_feats)
        debug["encoder"] = encoded
        fpn_feats = self._run_fpn(encoded)
        debug["fpn"] = fpn_feats
        pan_feats = self._run_pan(fpn_feats)
        debug["pan"] = pan_feats
        debug["final"] = pan_feats
        return pan_feats, debug

    # used for test against pytorch , builds 2D sine-cosine positional embeddings
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    # clean
    def close(self):
        if self._owns_device:
            try:
                self.ttnn.close_device(self.device)
            except Exception:
                pass
