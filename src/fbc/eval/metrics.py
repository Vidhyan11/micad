"""Evaluation metrics for diagnosis and concepts."""
from __future__ import annotations

import numpy as np


def dx_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score)
    pred = logits.argmax(1)
    probs = _softmax(logits)
    out = {
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "acc": float((pred == y_true).mean()),
    }
    try:
        if logits.shape[1] == 2:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            out["auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                               average="macro"))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def concept_metrics(targets: np.ndarray, mask: np.ndarray, probs: np.ndarray,
                    keys: list) -> dict:
    """Per-concept AUROC where mask==1 and the (binarized) target has both classes."""
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, k in enumerate(keys):
        m = mask[:, j] == 1
        if m.sum() < 10:
            continue
        y = (targets[m, j] >= 0.5).astype(int)
        if len(np.unique(y)) < 2:
            continue
        per[k] = float(roc_auc_score(y, probs[m, j]))
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return {"mean_auroc": mean, "per_concept": per}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
