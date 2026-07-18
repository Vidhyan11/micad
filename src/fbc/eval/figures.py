"""Qualitative figures: per-case concept explanation + the counterfactual concept
that flips the diagnosis (the 'faithful example' panel)."""
from __future__ import annotations

import numpy as np


def faithfulness_bar_chart(df, out_path):
    """Grouped bars: reliance & comprehensiveness (with 95% CI whiskers) for the
    pure bottleneck vs. the leaky model, per model. df = exp2_faithfulness.csv.

    Color = identity (2 categories): faithful pure bottleneck (blue) vs. leaky
    (muted grey) — a hue+lightness pair that is colorblind- and print-safe.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PURE, LEAKY = "#2c6fbb", "#9aa3a8"
    metrics = ["reliance", "comprehensiveness"]
    models = [m for m in ["A", "B"] if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.9 * len(models), 3.2),
                             constrained_layout=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        sub = df[df["model"] == m]
        pure = sub[sub["variant"] == "pure-bottleneck"].iloc[0]
        leak = sub[sub["variant"] == "leaky"].iloc[0]
        x = np.arange(len(metrics)); w = 0.36
        for row, color, lab, off in [(pure, PURE, "pure bottleneck", -w / 2),
                                     (leak, LEAKY, "leaky", w / 2)]:
            vals = [float(row[me]) for me in metrics]
            lo = [float(row[me]) - float(row[me + "_lo"]) for me in metrics]
            hi = [float(row[me + "_hi"]) - float(row[me]) for me in metrics]
            ax.bar(x + off, vals, w, color=color, label=lab, zorder=3,
                   yerr=[lo, hi], capsize=3, error_kw={"lw": 1, "ecolor": "#333"})
            for xi, v, h in zip(x + off, vals, hi):
                ax.text(xi, v + h + 0.009, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(["reliance", "compreh."], fontsize=8)
        ax.set_title(f"Model {m}", fontsize=9)
        ax.set_ylim(0, 0.28)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    axes[0].set_ylabel("faithfulness (95% CI)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
