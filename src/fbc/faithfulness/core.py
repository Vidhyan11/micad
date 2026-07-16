"""Concept-counterfactual faithfulness (contribution 1).

Given a model's diagnosis-from-concepts function and predicted concept probs, we
compare what the model SAYS it used (stated importance) against what actually
moves the decision (concept-space intervention). All operations are inference /
autograd only — no image generator.

Metrics (per test set):
  ccf_corr        mean per-case Spearman(|I_c|, |E_c|) — attribution fidelity
  decisive_hit    frac. of cases where argmax|I_c| == argmax|E_c|
  comprehensiveness  mean prob drop when ablating top-k STATED concepts (AOPC over k)
  sufficiency     mean prob drop when keeping ONLY top-k stated concepts (lower=better)
  reliance        mean prob drop when ALL concepts are ablated to reference
                  (how much the decision truly depends on concepts; ~1 for a pure
                  bottleneck, lower for a leaky model)

`predict_fn(concept_probs)->logits` closes over any fixed context (e.g. the
embedding for a leaky model), so pure-bottleneck and leaky models share this code.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

PredictFn = Callable[[torch.Tensor], torch.Tensor]


def _pred_prob(logits: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).gather(1, yhat[:, None]).squeeze(1)


def _rowwise_spearman(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Per-row Spearman correlation between (N,K) tensors."""
    ra = A.argsort(1).argsort(1).float()
    rb = B.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    num = (ra * rb).sum(1)
    den = torch.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-8
    return num / den


def stated_importance(concept_probs: torch.Tensor, predict_fn: PredictFn,
                      yhat: torch.Tensor, method: str = "gradient") -> torch.Tensor:
    """|attribution| of each concept to the predicted class. (N, K)."""
    if method == "gradient":
        c = concept_probs.clone().requires_grad_(True)
        p = _pred_prob(predict_fn(c), yhat)
        g = torch.autograd.grad(p.sum(), c)[0]
        return g.abs().detach()
    if method == "loo":                       # leave-one-out drop (to reference 0)
        return counterfactual_effect(concept_probs, predict_fn, yhat,
                                     mode="ablate", reference=torch.zeros(
                                         concept_probs.shape[1], device=concept_probs.device))
    raise ValueError(method)


def counterfactual_effect(concept_probs: torch.Tensor, predict_fn: PredictFn,
                          yhat: torch.Tensor, mode: str = "flip",
                          reference: torch.Tensor | None = None) -> torch.Tensor:
    """|Δ predicted-class prob| when each concept is intervened on. (N, K).

    mode='flip'   -> set concept c to (1 - c)  (present<->absent)
    mode='ablate' -> set concept c to reference value (ERASER-style)
    """
    N, K = concept_probs.shape
    p0 = _pred_prob(predict_fn(concept_probs), yhat)
    E = torch.zeros(N, K, device=concept_probs.device)
    for k in range(K):
        ck = concept_probs.clone()
        if mode == "flip":
            ck[:, k] = 1.0 - ck[:, k]
        elif mode == "ablate":
            ck[:, k] = 0.0 if reference is None else reference[k]
        else:
            raise ValueError(mode)
        pk = _pred_prob(predict_fn(ck), yhat)
        E[:, k] = (p0 - pk).abs()
    return E.detach()


def _aopc_persample(concept_probs, predict_fn, yhat, order, reference, keep: bool):
    """Per-sample average over k of prob-drop when ablating (keep=False) or keeping
    only (keep=True) the top-k concepts given by `order`. Returns (N,)."""
    N, K = concept_probs.shape
    p0 = _pred_prob(predict_fn(concept_probs), yhat)
    ref_full = reference[None, :].expand(N, K)
    acc = torch.zeros(N, device=concept_probs.device)
    for k in range(1, K):
        topk = torch.zeros(N, K, dtype=torch.bool, device=concept_probs.device)
        topk.scatter_(1, order[:, :k], True)
        c = torch.where(topk if not keep else ~topk, ref_full, concept_probs)
        pk = _pred_prob(predict_fn(c), yhat)
        acc += (p0 - pk).abs()
    return acc / max(K - 1, 1)


def faithfulness_scores(concept_probs: torch.Tensor, predict_fn: PredictFn,
                        reference: torch.Tensor, importance: str = "gradient",
                        cf_mode: str = "flip", return_per_sample: bool = False) -> dict:
    """Compute the full faithfulness metric bundle for one model on one split.

    If return_per_sample=True, also returns '_per_sample' with (N,) numpy arrays for
    each metric — enabling fast bootstrap CIs without recomputing interventions.
    """
    with torch.no_grad():
        logits0 = predict_fn(concept_probs)
        yhat = logits0.argmax(1)
        p0 = _pred_prob(logits0, yhat)

    I = stated_importance(concept_probs, predict_fn, yhat, method=importance)
    E = counterfactual_effect(concept_probs, predict_fn, yhat, mode=cf_mode,
                              reference=reference)

    ccf = _rowwise_spearman(I, E)                                   # (N,)
    decisive = (I.argmax(1) == E.argmax(1)).float()                 # (N,)
    order = I.argsort(1, descending=True)
    with torch.no_grad():
        comp = _aopc_persample(concept_probs, predict_fn, yhat, order, reference, keep=False)
        suff = _aopc_persample(concept_probs, predict_fn, yhat, order, reference, keep=True)
        c_all = reference[None, :].expand_as(concept_probs).clone()
        p_all = _pred_prob(predict_fn(c_all), yhat)
        reliance = (p0 - p_all).abs()                               # (N,)

    per = {
        "ccf_corr": ccf.detach().cpu().numpy(),
        "decisive_hit": decisive.detach().cpu().numpy(),
        "comprehensiveness": comp.detach().cpu().numpy(),
        "sufficiency": suff.detach().cpu().numpy(),
        "reliance": reliance.detach().cpu().numpy(),
    }
    out = {k: float(np.nanmean(v)) for k, v in per.items()}
    out["n"] = int(concept_probs.shape[0])
    if return_per_sample:
        out["_per_sample"] = per
    return out
