"""Concept vocabularies.

Faithful-by-Concept uses TWO concept vocabularies (see the two-model design):
  * DERMOSCOPIC — the 7-point checklist, ground-truthed by derm7pt (Model A).
  * CLINICAL    — ABCD + visible morphology, scoreable in clinical photos and
                  foundation-bootstrapped on Fitzpatrick17k/PAD-UFES (Model B).

Both are held in a single UNION list `CONCEPTS` (dermoscopic first, then clinical).
Every dataset row carries the full union of concept/mask columns (see schema.py);
per-dataset masks mark which concepts are annotated. Each model reads only its
own vocabulary subset (DERMOSCOPIC_KEYS or CLINICAL_KEYS).

Zero-shot scoring (MP) uses PROMPT ENSEMBLING: each concept has a short positive
and negative phrase, expanded over several domain templates and averaged — more
robust than a single sentence.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    pos_phrase: str      # bare phrase, filled into templates
    neg_phrase: str
    domain: str          # "dermoscopic" | "clinical"


# Prompt templates per domain (ensembled + averaged in encoders/clip_encoder.py).
TEMPLATES: dict[str, list[str]] = {
    "dermoscopic": [
        "dermoscopy of {}",
        "a dermoscopic image of {}",
        "dermoscopy showing {}",
        "a dermoscopic image with {}",
    ],
    "clinical": [
        "a photo of {}",
        "a clinical photograph of {}",
        "a skin lesion with {}",
        "a dermatology photo showing {}",
    ],
}


def prompt_sets(concept: Concept) -> tuple[list[str], list[str]]:
    """(positive_prompts, negative_prompts) for a concept, ensembled over templates."""
    tpls = TEMPLATES[concept.domain]
    pos = [t.format(concept.pos_phrase) for t in tpls]
    neg = [t.format(concept.neg_phrase) for t in tpls]
    return pos, neg


# --- Model A: dermoscopic 7-point checklist (derm7pt GT) -------------------- #
DERMOSCOPIC_CONCEPTS: list[Concept] = [
    Concept("pigment_network", "Pigment network",
            "an atypical pigment network", "a typical pigment network", "dermoscopic"),
    Concept("blue_whitish_veil", "Blue-whitish veil",
            "a blue-whitish veil", "no blue-whitish veil", "dermoscopic"),
    Concept("vascular_structures", "Vascular structures",
            "atypical vessels", "no atypical vessels", "dermoscopic"),
    Concept("pigmentation", "Pigmentation",
            "irregular pigmentation", "regular pigmentation", "dermoscopic"),
    Concept("streaks", "Streaks",
            "irregular streaks", "no streaks", "dermoscopic"),
    Concept("dots_globules", "Dots and globules",
            "irregular dots and globules", "regular dots and globules", "dermoscopic"),
    Concept("regression_structures", "Regression structures",
            "regression structures", "no regression structures", "dermoscopic"),
]

# --- Model B: clinical ABCD + morphology (foundation-bootstrapped) ---------- #
CLINICAL_CONCEPTS: list[Concept] = [
    Concept("asymmetry", "Asymmetry",
            "an asymmetric shape", "a symmetric round shape", "clinical"),
    Concept("border_irregularity", "Border irregularity",
            "an irregular ragged border", "a smooth well-defined border", "clinical"),
    Concept("color_variegation", "Color variegation",
            "several different colors", "one uniform color", "clinical"),
    # NOTE: 'large_diameter' was dropped — absolute size is not visually recoverable
    # from a lesion crop (no scale reference); DermLIP zero-shot AUROC ~0.55 vs
    # PAD-UFES diameter GT. Kept as a reported negative finding, not a bottleneck concept.
    Concept("elevation", "Elevation",
            "a raised bumpy surface", "a flat surface", "clinical"),
    Concept("ulceration", "Ulceration",
            "an open ulcer or bleeding", "intact unbroken skin", "clinical"),
    Concept("scale", "Scale/crust",
            "dry scale or crust", "a smooth surface without scale", "clinical"),
    Concept("erythema", "Erythema",
            "red inflamed skin", "normal skin color", "clinical"),
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
    # PAD-UFES metadata gives partial clinical GT (used to VALIDATE pseudo-labels):
    "pad_ufes_20": ["elevation"],
    "fitzpatrick17k": [],                 # clinical concepts via pseudo-labels only
}

# Which concept vocabulary each dataset/model operates in.
DATASET_DOMAIN: dict[str, str] = {
    "derm7pt": "dermoscopic",
    "fitzpatrick17k": "clinical",
    "pad_ufes_20": "clinical",
}
