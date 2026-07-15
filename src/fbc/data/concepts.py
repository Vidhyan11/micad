"""Concept vocabularies.

Faithful-by-Concept uses TWO concept vocabularies (see the two-model design):
  * DERMOSCOPIC — the 7-point checklist, ground-truthed by derm7pt (Model A).
  * CLINICAL    — ABCD + visible morphology, scoreable in clinical photos and
                  foundation-bootstrapped on Fitzpatrick17k/PAD-UFES (Model B).

Both are held in a single UNION list `CONCEPTS` (dermoscopic first, then clinical).
Every dataset row carries the full union of concept/mask columns (see schema.py);
per-dataset masks mark which concepts are annotated. Each model reads only its
own vocabulary subset (DERMOSCOPIC_KEYS or CLINICAL_KEYS).

Zero-shot pos/neg prompt pairs drive the foundation pseudo-labeler (MP).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    prompt_pos: str
    prompt_neg: str
    domain: str          # "dermoscopic" | "clinical"


# --- Model A: dermoscopic 7-point checklist (derm7pt GT) -------------------- #
DERMOSCOPIC_CONCEPTS: list[Concept] = [
    Concept("pigment_network", "Pigment network",
            "dermoscopy showing an atypical pigment network",
            "dermoscopy with a typical or absent pigment network", "dermoscopic"),
    Concept("blue_whitish_veil", "Blue-whitish veil",
            "dermoscopy showing a blue-whitish veil",
            "dermoscopy with no blue-whitish veil", "dermoscopic"),
    Concept("vascular_structures", "Vascular structures",
            "dermoscopy showing atypical vascular structures",
            "dermoscopy with no atypical vascular structures", "dermoscopic"),
    Concept("pigmentation", "Pigmentation",
            "dermoscopy showing irregular pigmentation",
            "dermoscopy with regular or absent pigmentation", "dermoscopic"),
    Concept("streaks", "Streaks",
            "dermoscopy showing irregular streaks",
            "dermoscopy with no streaks", "dermoscopic"),
    Concept("dots_globules", "Dots and globules",
            "dermoscopy showing irregular dots and globules",
            "dermoscopy with regular or absent dots and globules", "dermoscopic"),
    Concept("regression_structures", "Regression structures",
            "dermoscopy showing regression structures",
            "dermoscopy with no regression structures", "dermoscopic"),
]

# --- Model B: clinical ABCD + morphology (foundation-bootstrapped) ---------- #
CLINICAL_CONCEPTS: list[Concept] = [
    Concept("asymmetry", "Asymmetry",
            "an asymmetric skin lesion",
            "a symmetric skin lesion", "clinical"),
    Concept("border_irregularity", "Border irregularity",
            "a skin lesion with an irregular, poorly defined border",
            "a skin lesion with a regular, well-defined border", "clinical"),
    Concept("color_variegation", "Color variegation",
            "a skin lesion with multiple colors",
            "a skin lesion with a single uniform color", "clinical"),
    Concept("large_diameter", "Large diameter",
            "a large skin lesion",
            "a small skin lesion", "clinical"),
    Concept("elevation", "Elevation",
            "a raised or elevated skin lesion",
            "a flat skin lesion", "clinical"),
    Concept("ulceration", "Ulceration",
            "a skin lesion with ulceration or bleeding",
            "a skin lesion with intact surface", "clinical"),
    Concept("scale", "Scale/crust",
            "a skin lesion with scale or crust",
            "a smooth skin lesion without scale", "clinical"),
    Concept("erythema", "Erythema",
            "a red, inflamed skin lesion",
            "a skin lesion without redness", "clinical"),
]

# Union (dermoscopic first). Index in CONCEPTS == index in the concept vector.
CONCEPTS: list[Concept] = DERMOSCOPIC_CONCEPTS + CLINICAL_CONCEPTS

CONCEPT_KEYS: list[str] = [c.key for c in CONCEPTS]
NUM_CONCEPTS: int = len(CONCEPTS)
DERMOSCOPIC_KEYS: list[str] = [c.key for c in DERMOSCOPIC_CONCEPTS]
CLINICAL_KEYS: list[str] = [c.key for c in CLINICAL_CONCEPTS]

CONCEPT_BY_KEY: dict[str, Concept] = {c.key: c for c in CONCEPTS}


def index_of(key: str) -> int:
    return CONCEPT_KEYS.index(key)


def keys_for_domain(domain: str) -> list[str]:
    if domain == "dermoscopic":
        return DERMOSCOPIC_KEYS
    if domain == "clinical":
        return CLINICAL_KEYS
    raise ValueError(f"unknown domain {domain!r}")


def mask_for(available_keys: list[str]) -> list[int]:
    """1 where the dataset annotates the concept, else 0 — over the union order."""
    avail = set(available_keys)
    return [1 if k in avail else 0 for k in CONCEPT_KEYS]


# Which concepts each dataset provides as GT (others -> foundation pseudo-labels).
DATASET_GT_CONCEPTS: dict[str, list[str]] = {
    "derm7pt": DERMOSCOPIC_KEYS,          # real 7-pt GT
    "fitzpatrick17k": [],                 # clinical concepts via pseudo-labels
    "pad_ufes_20": [],                    # clinical concepts via pseudo-labels
}

# Which concept vocabulary each dataset/model operates in.
DATASET_DOMAIN: dict[str, str] = {
    "derm7pt": "dermoscopic",
    "fitzpatrick17k": "clinical",
    "pad_ufes_20": "clinical",
}
