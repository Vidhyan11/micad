"""CLIP-style derm encoders (DermLIP primary; MONET) via open_clip.

Besides image embeddings, exposes `zeroshot_concepts` — scoring each canonical
concept from a positive/negative text-prompt pair — used by the foundation
pseudo-labeler (MP) on datasets lacking concept ground truth.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..data.concepts import CONCEPTS
from .base import Encoder


class OpenClipEncoder(Encoder):
    """Generic open_clip loader via an 'hf-hub:<repo>' spec."""

    def __init__(self, name: str, hf_hub_spec: str, **kw):
        super().__init__(**kw)
        self.name = name
        self.hf_hub_spec = hf_hub_spec

    def _load(self) -> None:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(self.hf_hub_spec)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.hf_hub_spec)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self.dim = int(self.model.encode_image(dummy).shape[-1])

    @torch.no_grad()
    def encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feat = self.model.encode_image(batch)
        return feat.float().cpu().numpy()

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(prompts).to(self.device)
        t = self.model.encode_text(toks)
        return t / t.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def zeroshot_concepts(
        self, image_embeddings: np.ndarray, temperature: float = 0.01
    ) -> np.ndarray:
        """Score each canonical concept from its pos/neg prompt pair.

        Returns (N, K) probabilities: softmax over {neg, pos} similarity, pos column.
        Assumes `image_embeddings` are L2-normalized (as embed_paths produces).
        """
        img = torch.from_numpy(np.asarray(image_embeddings, dtype=np.float32)).to(self.device)
        img = img / img.norm(dim=-1, keepdim=True)
        probs = np.zeros((img.shape[0], len(CONCEPTS)), dtype=np.float32)
        for ci, c in enumerate(CONCEPTS):
            txt = self._encode_text([c.prompt_neg, c.prompt_pos])   # (2, d)
            logits = (img @ txt.T) / temperature                    # (N, 2)
            p = torch.softmax(logits, dim=-1)[:, 1]                 # P(pos)
            probs[:, ci] = p.cpu().numpy()
        return probs


def dermlip(**kw) -> OpenClipEncoder:
    # DermLIP ViT-B/16 (ICCV'25) — primary derm encoder.
    return OpenClipEncoder("dermlip", "hf-hub:redlessone/DermLIP_ViT-B-16", **kw)


def monet(**kw) -> OpenClipEncoder:
    # MONET (Nat. Med. 2024). NOTE: verify open_clip compatibility of this repo;
    # may require the official loading path if this fails.
    return OpenClipEncoder("monet", "hf-hub:suinleelab/monet", **kw)
