"""Losses: masked concept BCE (soft targets ok) + diagnosis CE."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over concepts, averaged only where mask==1.
    targets may be soft (pseudo-label probabilities) or hard {0,1}.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def diagnosis_ce(logits: torch.Tensor, y: torch.Tensor,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, y, weight=weight)
