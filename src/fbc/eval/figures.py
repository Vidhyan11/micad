"""Qualitative figures: per-case concept explanation + the counterfactual concept
that flips the diagnosis (the 'faithful example' panel)."""
from __future__ import annotations

import numpy as np


def concept_counterfactual_panel(concept_probs: np.ndarray, keys: list,
                                 effects: np.ndarray, pred_name: str,
                                 title: str, out_path):
    """Bar panel: predicted concept probabilities (left) and each concept's
    counterfactual effect on the decision (right); the top-effect concept is the
    one whose flip most changes the diagnosis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(effects)[::-1]
    keys = [keys[i] for i in order]
    probs = concept_probs[order]
    eff = effects[order]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    y = np.arange(len(keys))
    ax[0].barh(y, probs, color="#4c72b0")
    ax[0].set_yticks(y); ax[0].set_yticklabels(keys, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("predicted concept probability")
    ax[0].set_title(f"Concepts (pred: {pred_name})", fontsize=9)

    colors = ["#c44e52" if i == 0 else "#8c8c8c" for i in range(len(keys))]
    ax[1].barh(y, eff, color=colors)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel("Δ decision prob if flipped")
    ax[1].set_title("Counterfactual effect (red = decisive)", fontsize=9)

    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
