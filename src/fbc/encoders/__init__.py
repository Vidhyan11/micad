"""Frozen foundation encoders (DINOv2 reliable; DermLIP primary; MONET)."""
from __future__ import annotations

from .base import Encoder


def build_encoder(name: str, device: str = "cuda", **kw) -> Encoder:
    """Factory: build a frozen encoder by logical name."""
    name = name.lower()
    if name == "dinov2":
        from .dinov2 import DINOv2Encoder
        return DINOv2Encoder(device=device, **kw)
    if name == "dermlip":
        from .clip_encoder import dermlip
        return dermlip(device=device, **kw)
    if name == "monet":
        from .clip_encoder import monet
        return monet(device=device, **kw)
    raise KeyError(f"unknown encoder {name!r} (have: dinov2, dermlip, monet)")


__all__ = ["Encoder", "build_encoder"]
