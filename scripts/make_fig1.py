"""Render the Faithful-by-Concept architecture (paper Fig. 1) as a high-res raster,
matching the in-paper TikZ diagram, for the camera-ready 'separate figure files'
requirement. Saves 'Fig 1.jpg' at 300 DPI into D:/micad/submission/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = r"D:\micad\submission"
os.makedirs(OUT_DIR, exist_ok=True)

FROZEN = "#e9e9e9"
TRAINED = "#dce6f7"
TRAINED_EDGE = "#4a76c4"
INK = "#111111"
RED = "#c0392b"

fig, ax = plt.subplots(figsize=(10, 3.1))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")

# box centres (x), common y
xs = [0.95, 3.05, 5.15, 7.25, 9.35]
y = 1.95
bw, bh = 1.55, 0.95
labels = [
    ("Lesion\nimage $x$", "white", INK),
    ("Frozen\nencoder $f$", FROZEN, INK),
    ("Concept\nhead $g$", TRAINED, TRAINED_EDGE),
    ("Diagnosis\nhead $h$", TRAINED, TRAINED_EDGE),
    ("label\n$\\hat{y}$", "white", INK),
]
centres = {}
for x, (txt, fill, edge) in zip(xs, labels):
    box = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, txt, ha="center", va="center", fontsize=12, color=INK, zorder=4)
    centres[txt] = x

# arrows between boxes (with edge labels z and c-hat)
def arrow(x0, x1, yy=y):
    ax.add_patch(FancyArrowPatch((x0, yy), (x1, yy), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.8, color=INK, zorder=2))

for i in range(4):
    arrow(xs[i] + bw / 2, xs[i + 1] - bw / 2)
ax.text((xs[1] + xs[2]) / 2, y + 0.28, "$z$", ha="center", fontsize=12, style="italic")
ax.text((xs[2] + xs[3]) / 2, y + 0.28, "$\\hat{c}$", ha="center", fontsize=12, style="italic")

# frozen / trained captions
ax.text(xs[1], y - bh / 2 - 0.22, "frozen", ha="center", fontsize=9, style="italic", color="#555")
ax.text((xs[2] + xs[3]) / 2, y - bh / 2 - 0.22, "trained (small MLPs)",
        ha="center", fontsize=9, style="italic", color="#555")

# counterfactual callout below, dashed red, arrow up to diagnosis head
cx, cy, cw, ch = 5.15, 0.55, 4.7, 0.7
call = FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                      boxstyle="round,pad=0.02,rounding_size=0.10",
                      linewidth=1.4, edgecolor=RED, facecolor="#fdecea",
                      linestyle="--", zorder=3)
ax.add_patch(call)
ax.text(cx, cy, "Counterfactual test: flip a concept $\\hat{c}_c\\!\\to\\!1-\\hat{c}_c$, "
        "re-run $h$, measure $|\\Delta P(\\hat{y})|$",
        ha="center", va="center", fontsize=10.5, color=INK, zorder=4)
ax.add_patch(FancyArrowPatch((xs[3], cy + ch / 2), (xs[3], y - bh / 2),
             arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
             linestyle="--", color=RED, zorder=2))

fig.tight_layout()
out = os.path.join(OUT_DIR, "Fig 1.jpg")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved:", out, os.path.getsize(out), "bytes")
