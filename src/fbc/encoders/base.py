"""Frozen-encoder base class. Subclasses implement `_load` and `encode_pil_batch`;
the base handles image IO, batching, error handling, and L2 normalization.

All encoders are used with torch.no_grad and never trained.
"""
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


class Encoder(abc.ABC):
    name: str = "base"
    dim: int = 0

    def __init__(self, device: str = "cuda", l2_normalize: bool = True):
        self.device = device
        self.l2_normalize = l2_normalize
        self._loaded = False

    # --- subclass hooks ---------------------------------------------------- #
    @abc.abstractmethod
    def _load(self) -> None:
        """Load model + preprocessing onto self.device; set self.dim."""

    @abc.abstractmethod
    def encode_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL RGB images -> (B, dim) float32 (pre-normalization)."""

    # --- public API -------------------------------------------------------- #
    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def embed_paths(
        self,
        paths: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed images at `paths`. Returns (embeddings (N,dim) float32,
        valid_mask (N,) bool). Failed/missing images get a zero row + mask False.
        """
        self.ensure_loaded()
        n = len(paths)
        out = np.zeros((n, self.dim), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        it = range(0, n, batch_size)
        if show_progress:
            it = tqdm(it, desc=f"embed[{self.name}]", unit="batch")
        for start in it:
            chunk = list(paths[start:start + batch_size])
            imgs, idxs = [], []
            for j, p in enumerate(chunk):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    idxs.append(start + j)
                except Exception:  # noqa: BLE001 — missing/corrupt image
                    continue
            if not imgs:
                continue
            vecs = self.encode_pil_batch(imgs)
            vecs = np.asarray(vecs, dtype=np.float32)
            if self.l2_normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.clip(norms, 1e-8, None)
            for k, gi in enumerate(idxs):
                out[gi] = vecs[k]
                valid[gi] = True
        return out, valid
