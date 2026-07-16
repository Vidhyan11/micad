"""Group-conditional concept calibration (contribution 3 mitigation).

If concept predictions are miscalibrated differently across skin-tone groups, the
downstream reasoning (and its faithfulness) can be inequitable. We fit a per-group
temperature on the concept logits (on a held-out calibration split), so concept
probabilities are comparably reliable across groups, then re-audit.
"""
from __future__ import annotations

import numpy as np
import torch


def fit_group_temperature(concept_logits: torch.Tensor, targets: torch.Tensor,
                          groups: np.ndarray, group_names: list,
                          iters: int = 200) -> dict:
    """Fit one temperature T_g per group minimizing concept BCE(sigmoid(logit/T), target).
    Returns {group_name: T}. Groups with too few samples default to T=1."""
    temps = {}
    for g in group_names:
        m = groups == g
        if m.sum() < 20:
            temps[g] = 1.0
            continue
        logit_g = concept_logits[m]
        tgt_g = targets[m]
        logT = torch.zeros(1, requires_grad=True, device=concept_logits.device)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)

        def closure():
            opt.zero_grad()
            T = logT.exp()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit_g / T, tgt_g)
            loss.backward()
            return loss

        opt.step(closure)
        temps[g] = float(logT.exp().item())
    return temps


def apply_group_temperature(concept_logits: torch.Tensor, groups: np.ndarray,
                            temps: dict) -> torch.Tensor:
    """Return calibrated concept PROBS = sigmoid(logit / T_group)."""
    T = torch.ones(concept_logits.shape[0], 1, device=concept_logits.device)
    for g, t in temps.items():
        m = torch.as_tensor(groups == g, device=concept_logits.device)
        if m.any():
            T[m] = t
    return torch.sigmoid(concept_logits / T)
