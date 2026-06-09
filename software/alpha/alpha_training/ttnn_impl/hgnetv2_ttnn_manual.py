
"""TTNN reimplementation of the HGNetv2 backbone used by D-FINE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import ttnn


try:  # normalise runtime flags once at import time, 
    ttnn.CONFIG.enable_fast_runtime_mode = False
    ttnn.CONFIG.enable_model_cache = True # performance issues ? 
except Exception:
    pass


@dataclass
class TTActivation:
    tensor: "ttnn.Tensor"
    batch: int
    height: int
    width: int
    channels: int

# on host or TT ? 
def fold_bn_to_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fused (weight, bias) for conv2d + batchnorm."""
    W = conv.weight.detach().clone()  # [Cout, Cin/groups, Kh, Kw]
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(W.shape[0], dtype=W.dtype)

    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()
    eps = bn.eps

    denom = torch.sqrt(var + eps)
    scale = (gamma / denom).reshape(-1, 1, 1, 1)

    W_fused = W * scale
    b_fused = beta + (b - mean) * gamma / denom
    return W_fused, b_fused


def _torch_to_tt_activation(
    x: torch.Tensor, device, dtype, layout
) -> TTActivation:
    n, c, h, w = x.shape
    nhwc = x.permute(0, 2, 3, 1).contiguous()
    flat = nhwc.reshape(1, 1, n * h * w, c)
    tt_tensor = ttnn.from_torch(flat.detach(), dtype=dtype, layout=layout, device=device)
    return TTActivation(tt_tensor, n, h, w, c)


def _tt_activation_to_torch(act: TTActivation) -> torch.Tensor:
    # Bring to host first to avoid device-side reshape constraints
    torch_flat = ttnn.to_torch(act.tensor)
    nhwc_shape = (act.batch, act.height, act.width, act.channels)
    torch_nhwc = torch_flat.reshape(nhwc_shape)
    return torch_nhwc.permute(0, 3, 1, 2).contiguous()


def _activation_to_nhwc(act: TTActivation):
    return ttnn.reshape(act.tensor, (act.batch, act.height, act.width, act.channels))


def _activation_from_nhwc(tensor, batch: int, height: int, width: int, channels: int) -> TTActivation:
    flat = ttnn.reshape(tensor, (1, 1, batch * height * width, channels))
    return TTActivation(flat, batch, height, width, channels)


def _activation_to_nchw(act: TTActivation):
    nhwc = _activation_to_nhwc(act)
    return ttnn.permute(nhwc, (0, 3, 1, 2))


def _activation_from_nchw(tensor, batch: int, channels: int, height: int, width: int) -> TTActivation:
    nhwc = ttnn.permute(tensor, (0, 2, 3, 1))
    return _activation_from_nhwc(nhwc, batch, height, width, channels)


def _pad_activation(act: TTActivation, pads: Tuple[int, int, int, int], value: float = 0.0) -> TTActivation:
    left, right, top, bottom = pads
    nhwc = _activation_to_nhwc(act)
    original_layout = nhwc.get_layout()
    nhwc_row_major = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT)
    padding = [(0, 0), (top, bottom), (left, right), (0, 0)]
    padded_rm = ttnn.pad(nhwc_row_major, padding=padding, value=value)
    padded = ttnn.to_layout(padded_rm, original_layout)
    return _activation_from_nhwc(
        padded, act.batch, act.height + top + bottom, act.width + left + right, act.channels
    )


def _concat_activations(acts: Sequence[TTActivation]) -> TTActivation:
    if not acts:
        raise ValueError("concat requires non-empty sequence")
    base = acts[0]
    nhwc_tensors = []
    for act in acts:
        nhwc = _activation_to_nhwc(act)
        # Keep ROW_MAJOR + DRAM for reliable concat behavior
        nhwc = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT)
        nhwc = ttnn.to_memory_config(nhwc, ttnn.DRAM_MEMORY_CONFIG)
        nhwc_tensors.append(nhwc)
    concatenated_nhwc = ttnn.concat(nhwc_tensors, dim=-1)
    channels = sum(act.channels for act in acts)
    return _activation_from_nhwc(concatenated_nhwc, base.batch, base.height, base.width, channels)


def _broadcast_spatial(act: TTActivation, target_height: int, target_width: int) -> TTActivation:
    if act.height == target_height and act.width == target_width:
        return act
    nhwc = _activation_to_nhwc(act)
    repeats = (1, target_height // act.height, target_width // act.width, 1)
    expanded = ttnn.repeat(nhwc, repeats)
    return _activation_from_nhwc(expanded, act.batch, target_height, target_width, act.channels)


def _pool_output_dim(
    input_size: int, kernel: int, stride: int, padding: int, dilation: int = 1, ceil_mode: bool = False
) -> int:
    numerator = input_size + 2 * padding - dilation * (kernel - 1) - 1
    if ceil_mode:
        numerator += stride - 1
    return numerator // stride + 1


def _conv_output_dim(input_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    numerator = input_size + 2 * padding - dilation * (kernel - 1) - 1
    return numerator // stride + 1


def _normalize_padding(padding) -> Tuple[int, ...]:
    if isinstance(padding, tuple):
        return tuple(int(p) for p in padding)
    if isinstance(padding, int):
        return (int(padding), int(padding))
    raise ValueError(f"Unsupported padding type: {padding!r}")


def _split_conv_module(conv_module: nn.Module) -> Tuple[nn.Conv2d, Optional[Tuple[int, int, int, int]]]:
    if isinstance(conv_module, nn.Conv2d):
        return conv_module, None
    if isinstance(conv_module, nn.Sequential):
        conv = None
        pad: Optional[Tuple[int, int, int, int]] = None
        for mod in conv_module:
            if isinstance(mod, nn.Conv2d):
                conv = mod
            elif isinstance(mod, nn.ZeroPad2d):
                pad = tuple(mod.padding)  # (left, right, top, bottom)
        if conv is None:
            raise ValueError("Sequential conv module missing Conv2d")
        return conv, pad
    raise TypeError(f"Unsupported conv container: {type(conv_module)}")


class TTNNConv2d(nn.Module):
    """TTNN Conv wrapper with pre-fused weights and bias (tile layout)."""

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, ...] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        dtype=None,
        layout=None,
        activation: Optional[str] = None,
    ):
        super().__init__()
        ttnn_mod = ttnn
        self.ttnn = ttnn_mod
        self.device = device
        self.stride = tuple(int(s) for s in stride)
        self.padding = tuple(int(p) for p in padding)
        self.dilation = tuple(int(d) for d in dilation)
        self.groups = int(groups)
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        # Prefer ROW_MAJOR layout for conv outputs to avoid tile sharding constraints
        self.layout = layout if layout is not None else ttnn.ROW_MAJOR_LAYOUT
        self.activation = activation

        weight = weight.detach().clone()
        # Reshape bias to 4D NHWC [1,1,1,Cout] as expected by TTNN conv2d
        bias = bias.detach().clone().reshape(1, 1, 1, -1)
        self.in_channels = weight.shape[1] * self.groups
        self.out_channels = weight.shape[0]
        self.kernel_size = tuple(weight.shape[-2:])

        if len(self.padding) == 2:
            self.pad_hw = (self.padding[0], self.padding[1])
        elif len(self.padding) == 4:
            self.pad_hw = (self.padding[0], self.padding[2])
        else:
            raise ValueError(f"Unsupported padding spec {self.padding}")

        # Host weights must be ROW_MAJOR for TTNN to prepare them for device
        self.weight = ttnn_mod.from_torch(
            weight, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.bias = ttnn_mod.from_torch(
            bias, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight.dtype,
            output_layout=self.layout,
        )
        self.compute_config = None
        # Output memory config (can be switched to L1 for speed where safe)
        self.memory_config = self.ttnn.DRAM_MEMORY_CONFIG
        # Track whether weights have been prepared on device to avoid reprocessing
        self._weights_prepared = False

    def use_l1_output(self):
        """Enable L1 output memory for this conv (faster, riskier)."""
        self.memory_config = self.ttnn.L1_MEMORY_CONFIG

    def forward(self, act: TTActivation) -> TTActivation:
        # Feed NHWC into conv2d (as in TTNN tutorials) and let conv apply padding
        x = act
        x_nhwc = _activation_to_nhwc(x)
        pad_for_conv = self.padding

        try:
            result = self.ttnn.conv2d(
                input_tensor=x_nhwc,
                weight_tensor=self.weight,
                bias_tensor=self.bias,  # apply folded bias during convolution
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=x.batch,
                input_height=x.height,
                input_width=x.width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=pad_for_conv,
                dilation=self.dilation,
                groups=self.groups,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                memory_config=self.memory_config,
                return_output_dim=False,
                return_weights_and_bias=not self._weights_prepared,
                dtype=self.dtype,
            )
            # result can be either tensor or (tensor, (H,W)) or (tensor, (weights,bias))
            if isinstance(result, tuple) and len(result) == 2:
                output_tensor, meta = result
                # meta may be (H, W) or (weights, bias). Detect tensors
                if (
                    isinstance(meta, tuple)
                    and len(meta) == 2
                    and hasattr(meta[0], "get_layout")
                ):
                    w_new, b_new = meta
                    if w_new is not None:
                        self.weight = w_new
                        self._weights_prepared = True
                    if b_new is not None:
                        self.bias = b_new
            else:
                output_tensor = result
            # Explicitly ensure output is in flat [1,1,NHW,C] form to keep pipeline consistent

            out_h = _conv_output_dim(x.height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0])
            out_w = _conv_output_dim(x.width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1])
            output_tensor = self.ttnn.reshape(output_tensor, (1, 1, x.batch * out_h * out_w, self.out_channels))
            # But perf effects on Tilization ? 
            # Bias is applied inside conv2d via bias_tensor; do not add again.
        except Exception as e:
            raise RuntimeError(
                (
                    "TTNNConv2d failed with shapes: "
                    f"in=[N={act.batch},H={act.height},W={act.width},C={act.channels}], "
                    f"weight=[OC={self.out_channels},IC_per_group={self.in_channels // max(self.groups,1)},KH={self.kernel_size[0]},KW={self.kernel_size[1]}], "
                    f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}"
                )
            ) from e
        out_height = _conv_output_dim(
            act.height, self.kernel_size[0], self.stride[0], self.pad_hw[0], self.dilation[0]
        )
        out_width = _conv_output_dim(
            act.width, self.kernel_size[1], self.stride[1], self.pad_hw[1], self.dilation[1]
        )
        return TTActivation(output_tensor, act.batch, out_height, out_width, self.out_channels)


class TTNNLearnableAffine(nn.Module):
    def __init__(self, lab_pt: nn.Module):
        super().__init__()
        self.scale = float(lab_pt.scale.detach().item())
        self.bias = float(lab_pt.bias.detach().item())

    def forward(self, act: TTActivation) -> TTActivation:
        scaled = ttnn.multiply(act.tensor, self.scale)
        shifted = ttnn.add(scaled, self.bias)
        return TTActivation(shifted, act.batch, act.height, act.width, act.channels)


class TTNNConvBNAct(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout):
        super().__init__()
        self.ttnn = ttnn
        conv_pt, explicit_pad = _split_conv_module(module_pt.conv)
        weight, bias = fold_bn_to_conv(conv_pt, module_pt.bn)
        padding = _normalize_padding(conv_pt.padding)
        self.conv = TTNNConv2d(
            device,
            weight,
            bias,
            stride=tuple(conv_pt.stride),
            padding=padding,
            dilation=tuple(conv_pt.dilation),
            groups=conv_pt.groups,
            dtype=dtype,
            layout=layout,
        )
        self.pre_pad = explicit_pad
        self.use_act = module_pt.use_act
        self.use_lab = module_pt.use_act and getattr(module_pt, "use_lab", False)
        self.lab = TTNNLearnableAffine(module_pt.lab) if self.use_lab else None

    def enable_l1(self):
        # Opt-in to faster L1 outputs when safe
        if hasattr(self, "conv") and hasattr(self.conv, "use_l1_output"):
            self.conv.use_l1_output()

    def forward(self, act: TTActivation) -> TTActivation:
        x = act
        if self.pre_pad is not None:
            x = _pad_activation(x, self.pre_pad)
        x = self.conv(x)
        if self.use_act:
            activated = self.ttnn.relu(x.tensor)
            x = TTActivation(activated, x.batch, x.height, x.width, x.channels)
        if self.lab is not None:
            x = self.lab(x)
        return x


class TTNNLightConvBNAct(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout):
        super().__init__()
        self.conv1 = TTNNConvBNAct(module_pt.conv1, device, dtype, layout)
        self.conv2 = TTNNConvBNAct(module_pt.conv2, device, dtype, layout)

    def forward(self, act: TTActivation) -> TTActivation:
        x = self.conv1(act)
        x = self.conv2(x)
        return x


class TTNNEseModule(nn.Module):
    def __init__(self, module_pt: nn.Module, device, dtype, layout):
        super().__init__()
        conv_pt = module_pt.conv
        weight = conv_pt.weight.detach().clone()
        bias = conv_pt.bias.detach().clone()
        self.conv = TTNNConv2d(
            device,
            weight,
            bias,
            stride=tuple(conv_pt.stride),
            padding=_normalize_padding(conv_pt.padding),
            dilation=tuple(conv_pt.dilation),
            groups=conv_pt.groups,
            dtype=dtype,
            layout=layout,
        )

    def forward(self, act: TTActivation) -> TTActivation:
        identity = act
        nchw = _activation_to_nchw(act)
        pooled = ttnn.global_avg_pool2d(nchw)
        pooled_act = _activation_from_nchw(pooled, act.batch, act.channels, 1, 1)
        gating = self.conv(pooled_act)
        gate_tensor = ttnn.sigmoid(gating.tensor)
        gating = TTActivation(gate_tensor, gating.batch, gating.height, gating.width, gating.channels)
        gating = _broadcast_spatial(gating, identity.height, identity.width)
        scaled = ttnn.multiply(identity.tensor, gating.tensor)
        return TTActivation(scaled, identity.batch, identity.height, identity.width, identity.channels)


class TTNNHGBlock(nn.Module):
    def __init__(self, block_pt: nn.Module, device, dtype, layout):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_pt in block_pt.layers:
            if hasattr(layer_pt, "conv1") and hasattr(layer_pt, "conv2"):
                self.layers.append(TTNNLightConvBNAct(layer_pt, device, dtype, layout))
            else:
                self.layers.append(TTNNConvBNAct(layer_pt, device, dtype, layout))

        self.aggregation = nn.ModuleList()
        for agg_mod in block_pt.aggregation:
            if isinstance(agg_mod, nn.Identity):
                continue
            if hasattr(agg_mod, "conv") and hasattr(agg_mod, "bn"):
                self.aggregation.append(TTNNConvBNAct(agg_mod, device, dtype, layout))
            else:
                self.aggregation.append(TTNNEseModule(agg_mod, device, dtype, layout))

        self.residual = block_pt.residual

    def forward(self, act: TTActivation) -> TTActivation:
        identity = act
        outputs = [act]
        x = act
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = _concat_activations(outputs)
        for module in self.aggregation:
            x = module(x)
        if self.residual:
            combined = ttnn.add(identity.tensor, x.tensor)
            x = TTActivation(combined, x.batch, x.height, x.width, x.channels)
        return x


class TTNNHGStage(nn.Module):
    def __init__(self, stage_pt: nn.Module, device, dtype, layout):
        super().__init__()
        self.downsample_op: Optional[nn.Module]
        if isinstance(stage_pt.downsample, nn.Identity):
            self.downsample_op = None
        else:
            self.downsample_op = TTNNConvBNAct(stage_pt.downsample, device, dtype, layout)
        self.blocks = nn.ModuleList(
            [TTNNHGBlock(block_pt, device, dtype, layout) for block_pt in stage_pt.blocks]
        )

    def forward(self, act: TTActivation) -> TTActivation:
        x = act
        if self.downsample_op is not None:
            x = self.downsample_op(x)
        for block in self.blocks:
            x = block(x)
        return x


class StemTTNN(nn.Module):
    """TTNN implementation of StemBlock in HGNetv2."""

    def __init__(self, stem_pt: nn.Module, device, dtype, layout):
        super().__init__()
        self.stem1 = TTNNConvBNAct(stem_pt.stem1, device, dtype, layout)
        self.stem2a = TTNNConvBNAct(stem_pt.stem2a, device, dtype, layout)
        self.stem2b = TTNNConvBNAct(stem_pt.stem2b, device, dtype, layout)
        self.stem3 = TTNNConvBNAct(stem_pt.stem3, device, dtype, layout)
        self.stem4 = TTNNConvBNAct(stem_pt.stem4, device, dtype, layout)
        self.pool_kernel = (2, 2)
        self.pool_stride = (1, 1)
        self.pool_padding = (0, 0)
        # L1 outputs can be enabled selectively after performance validation

    def forward(self, act: TTActivation) -> TTActivation:
        x = self.stem1(act)

        y_pad = _pad_activation(x, (0, 1, 0, 1))
        x2 = self.stem2a(y_pad)
        x2 = _pad_activation(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)

        # Pool the padded tensor to mirror PyTorch StemBlock behavior
        x1 = self._max_pool(y_pad)
        x = _concat_activations([x1, x2])

        x = self.stem3(x)
        x = self.stem4(x)
        return x

    def _max_pool(self, act: TTActivation) -> TTActivation:
        kernel = self.pool_kernel
        stride = self.pool_stride
        padding = self.pool_padding
        pooled = ttnn.max_pool2d(
            input_tensor=act.tensor,
            batch_size=act.batch,
            input_h=act.height,
            input_w=act.width,
            channels=act.channels,
            kernel_size=list(kernel),
            stride=list(stride),
            padding=list(padding),
            dilation=[1, 1],
            ceil_mode=True,
        )
        # Normalize memory + layout to avoid concat/sharding issues downstream
        pooled = ttnn.to_layout(pooled, ttnn.ROW_MAJOR_LAYOUT)
        pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)
        out_h = _pool_output_dim(act.height, kernel[0], stride[0], padding[0], ceil_mode=True)
        out_w = _pool_output_dim(act.width, kernel[1], stride[1], padding[1], ceil_mode=True)
        pooled_flat = ttnn.reshape(pooled, (1, 1, act.batch * out_h * out_w, act.channels))
        return TTActivation(pooled_flat, act.batch, out_h, out_w, act.channels)


class HGNetv2TTNNManual(nn.Module):
    """TTNN implementation of the HGNetv2 backbone."""

    def __init__(self, backbone_pt: nn.Module, device_id: int = 0):
        super().__init__()
        ttnn_mod = ttnn
        self.ttnn = ttnn_mod
        # Open device with a larger L1 small buffer partition to avoid tiny L1_SMALL OOMs during conv halo/config
        try:
            # Increase L1 small buffer partition to avoid OOM on large inputs (e.g., 640x640)
            self.device = ttnn_mod.open_device(device_id=device_id, l1_small_size=655360)
        except TypeError:
            # Fallback: older TTNN signature 
            self.device = ttnn_mod.open_device(device_id=device_id) # not used anymore ? 
        self.dtype = ttnn_mod.bfloat16
        self.layout = ttnn_mod.TILE_LAYOUT

        self.stem = StemTTNN(backbone_pt.stem, self.device, self.dtype, self.layout)
        self.stages = nn.ModuleList(
            [TTNNHGStage(stage_pt, self.device, self.dtype, self.layout) for stage_pt in backbone_pt.stages]
        )
        self.return_idx = tuple(backbone_pt.return_idx)

    def _to_ttnn(self, x: torch.Tensor) -> TTActivation:
        return _torch_to_tt_activation(x, self.device, self.dtype, self.layout)

    def stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        act = self._to_ttnn(x)
        out = self.stem(act)
        return _tt_activation_to_torch(out)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        act = self._to_ttnn(x)
        outputs: List[torch.Tensor] = []
        act = self.stem(act)
        for idx, stage in enumerate(self.stages):
            act = stage(act)
            if idx in self.return_idx:
                outputs.append(_tt_activation_to_torch(act))
        return outputs

    def collect_stage_outputs(self, x: torch.Tensor, include_stem: bool = True) -> List[torch.Tensor]:
        """Return torch tensors for stem and each stage for debugging."""
        act = self._to_ttnn(x)
        collected: List[torch.Tensor] = []
        act = self.stem(act)
        if include_stem:
            collected.append(_tt_activation_to_torch(act))
        for stage in self.stages:
            act = stage(act)
            collected.append(_tt_activation_to_torch(act))
        return collected

    def close(self):
        try:
            self.ttnn.close_device(self.device)
        except Exception:
            pass
