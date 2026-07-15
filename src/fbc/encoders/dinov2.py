"""DINOv2 encoder (general-purpose visual backbone, fallback / ablation).
Loads via HuggingFace transformers — reliable, no custom code needed.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import Encoder


class DINOv2Encoder(Encoder):
    name = "dinov2"

    def __init__(self, hf_id: str = "facebook/dinov2-base", **kw):
        super().__init__(**kw)
        self.hf_id = hf_id

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size          # 768 for dinov2-base

    @torch.no_grad()
    def encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        # DINOv2 pooler_output = CLS token representation
        feat = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0]
        return feat.float().cpu().numpy()
