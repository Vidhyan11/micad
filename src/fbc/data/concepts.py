"""Canonical concept vocabulary shared across all datasets.

Anchored on the 7-point checklist (derm7pt is our ground-truth source) plus the
ABCD 'asymmetry' criterion that PH2 provides. Every dataset maps into THIS fixed,
ordered space; concepts a dataset does not annotate are marked unavailable via a
mask so training/eval can ignore them per-sample.

Zero-shot text prompts (pos/neg pairs) are used by the foundation-model
pseudo-labeler (encoders/monet.py, encoders/dermlip.py) for datasets without GT.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    key: str                      # stable identifier used in code & columns
    name: str                     # human-readable (figures, tables)
    prompt_pos: str               # zero-shot positive prompt
    prompt_neg: str               # zero-shot negative prompt


# Ordered canonical vocabulary. Index in this list == index in the concept vector.
CONCEPTS: list[Concept] = [
    Concept("pigment_network", "Pigment network",
            "dermoscopy showing an atypical pigment network",
            "dermoscopy with a typical or absent pigment network"),
    Concept("blue_whitish_veil", "Blue-whitish veil",
            "dermoscopy showing a blue-whitish veil",
            "dermoscopy with no blue-whitish veil"),
    Concept("vascular_structures", "Vascular structures",
            "dermoscopy showing atypical vascular structures",
            "dermoscopy with no atypical vascular structures"),
    Concept("pigmentation", "Pigmentation",
            "dermoscopy showing irregular pigmentation",
            "dermoscopy with regular or absent pigmentation"),
    Concept("streaks", "Streaks",
            "dermoscopy showing irregular streaks",
            "dermoscopy with no streaks"),
    Concept("dots_globules", "Dots and globules",
            "dermoscopy showing irregular dots and globules",
            "dermoscopy with regular or absent dots and globules"),
    Concept("regression_structures", "Regression structures",
            "dermoscopy showing regression structures",
            "dermoscopy with no regression structures"),
    Concept("asymmetry", "Asymmetry",
            "an asymmetric skin lesion",
            "a symmetric skin lesion"),
]

CONCEPT_KEYS: list[str] = [c.key for c in CONCEPTS]
NUM_CONCEPTS: int = len(CONCEPTS)


def index_of(key: str) -> int:
    return CONCEPT_KEYS.index(key)


def concept_vector_template() -> list[float]:
    """A zero vector of the canonical length (fill per-sample)."""
    return [0.0] * NUM_CONCEPTS


def mask_for(available_keys: list[str]) -> list[int]:
    """1 where the dataset annotates the concept, else 0 — for masked losses."""
    avail = set(available_keys)
    return [1 if k in avail else 0 for k in CONCEPT_KEYS]


# Which concepts each dataset provides as ground truth (others -> pseudo-labels).
DATASET_GT_CONCEPTS: dict[str, list[str]] = {
    "derm7pt": [
        "pigment_network", "blue_whitish_veil", "vascular_structures",
        "pigmentation", "streaks", "dots_globules", "regression_structures",
    ],
    "ph2": [
        "pigment_network", "blue_whitish_veil", "streaks",
        "dots_globules", "regression_structures", "asymmetry",
    ],
    "fitzpatrick17k": [],       # no concept GT -> foundation pseudo-labels
    "pad_ufes_20": [],          # no dermoscopic concept GT -> pseudo-labels
}
