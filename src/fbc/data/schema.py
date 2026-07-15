"""Unified record schema every loader emits, so downstream code (embedding
extraction, training, faithfulness, fairness) is dataset-agnostic.

A loader returns a pandas DataFrame with these columns:

    uid           str    globally-unique id ("<source>_<n>")
    source        str    dataset logical name
    image_path    str    absolute path to the PRIMARY image to embed
    image_path_alt str   optional secondary view ("" if none)
    dx_name       str    grouped diagnosis name
    dx_label      int    grouped diagnosis index (stable per dataset)
    fitzpatrick   float  1..6 or NaN
    fst_group     str    "I-II"/"III-IV"/"V-VI" or "" (unknown)
    patient_id    str    grouping key for leakage-clean splits
    lesion_id     str    lesion grouping key
    split         str    "train"/"val"/"test" or "" (assigned later)

    c_<concept>   float  concept value in [0,1] (GT 0/1, or pseudo-prob); NaN if unknown
    m_<concept>   int    1 if this concept is annotated (GT) for the row, else 0

Concept/mask columns exist for every key in fbc.data.concepts.CONCEPT_KEYS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .concepts import CONCEPT_KEYS

BASE_COLUMNS = [
    "uid", "source", "image_path", "image_path_alt",
    "dx_name", "dx_label", "fitzpatrick", "fst_group",
    "patient_id", "lesion_id", "split",
]


def concept_col(key: str) -> str:
    return f"c_{key}"


def mask_col(key: str) -> str:
    return f"m_{key}"


CONCEPT_COLUMNS = [concept_col(k) for k in CONCEPT_KEYS]
MASK_COLUMNS = [mask_col(k) for k in CONCEPT_KEYS]
ALL_COLUMNS = BASE_COLUMNS + CONCEPT_COLUMNS + MASK_COLUMNS


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLUMNS)


def concepts_matrix(df: pd.DataFrame) -> np.ndarray:
    """(N, K) concept values in canonical order (NaN where unknown)."""
    return df[CONCEPT_COLUMNS].to_numpy(dtype=np.float32)


def masks_matrix(df: pd.DataFrame) -> np.ndarray:
    """(N, K) 0/1 concept-availability masks in canonical order."""
    return df[MASK_COLUMNS].to_numpy(dtype=np.float32)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all schema columns exist and ordering is canonical."""
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in CONCEPT_COLUMNS else (
                0 if col in MASK_COLUMNS else "")
    return df[ALL_COLUMNS]
