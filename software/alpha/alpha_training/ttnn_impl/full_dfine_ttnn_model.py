from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .hgnetv2_ttnn_manual import HGNetv2TTNNManual
from .hybrid_encoder_ttnn import HybridEncoderTTNN
from .dfine_decoder_ttnn import DFINETransformerTTNN
from src.zoo.dfine.postprocessor import DFINEPostProcessor


class DFINE_TTNN(nn.Module):
    def __init__(self, model_pt: nn.Module, device_id: int = 0):
        super().__init__()
        self.model_pt = model_pt.eval()
        self.backbone_tt = HGNetv2TTNNManual(self.model_pt.backbone.eval(), device_id=device_id)
        self.encoder_tt = HybridEncoderTTNN(self.model_pt.encoder.eval(), device=self.backbone_tt.device, return_stage="final")
        self.decoder_tt = DFINETransformerTTNN(self.model_pt.decoder.eval(), device=self.backbone_tt.device)

        # Build a deploy-mode postprocessor mirroring PyTorch pipeline
        num_classes = getattr(self.model_pt.decoder, "num_classes", 80)
        num_top = getattr(self.model_pt.decoder, "num_queries", 300)
        self.post = DFINEPostProcessor(num_classes=num_classes, num_top_queries=num_top).deploy()

        # Default pre-processing
        self.input_size = (512, 512)
        self.pre = T.Compose([T.Resize(self.input_size), T.ToTensor()])
        # Ensure dynamic positional embeddings for arbitrary input sizes
        try:
            if hasattr(self.model_pt, 'encoder') and hasattr(self.model_pt.encoder, 'eval_spatial_size'):
                self.model_pt.encoder.eval_spatial_size = None
            if hasattr(self.model_pt, 'decoder') and hasattr(self.model_pt.decoder, 'eval_spatial_size'):
                self.model_pt.decoder.eval_spatial_size = None
        except Exception:
            pass

    @torch.no_grad()
    def forward(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        *,
        score_threshold: float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        # Pre-process
        if isinstance(image, str):
            im = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            im = image.convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Assume already [B,C,H,W] tensor in 0..1
            x = image
            w = image.shape[-1]
            h = image.shape[-2]
            orig_sizes = torch.tensor([[w, h]], dtype=torch.float32)
        else:
            raise TypeError("Unsupported image input type")

        if not isinstance(image, torch.Tensor):
            w, h = im.size
            orig_sizes = torch.tensor([[w, h]], dtype=torch.float32)
            x = self.pre(im).unsqueeze(0)

        # Backbone + encoder + decoder
        feats = self.backbone_tt(x)
        feats = self.encoder_tt(feats)
        raw = self.decoder_tt(feats)

        # Post-process to pixel boxes and top queries
        labels, boxes, scores = self.post(raw, orig_sizes)

        # Filter by score threshold
        mask = scores[0] > score_threshold
        return {
            "pred_logits": raw.get("pred_logits", None),
            "pred_boxes": raw.get("pred_boxes", None),
            "labels": labels[0][mask],
            "boxes": boxes[0][mask],
            "scores": scores[0][mask],
        }

    def close(self):
        try:
            self.decoder_tt.close()
            self.encoder_tt.close()
            self.backbone_tt.close()
        except Exception:
            pass
