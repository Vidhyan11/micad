"""derm7pt loader — the PRIMARY dataset (true 7-point-checklist concept GT).

Emits the unified schema (see schema.py). Concepts are encoded as BINARY presence
of the *suspicious* variant, aligned with the melanoma 7-point checklist (the
variant that scores points). This binary encoding is what makes the
concept-counterfactual test (flip present<->absent) well-defined.

Diagnosis is collapsed to the standard 5 derm7pt categories: BCC / MEL / NEV / SK / MISC.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config
from . import concepts as C
from . import schema as S

# derm7pt meta.csv column -> our canonical concept key
DERM7PT_COL = {
    "pigment_network": "pigment_network",
    "blue_whitish_veil": "blue_whitish_veil",
    "vascular_structures": "vascular_structures",
    "pigmentation": "pigmentation",
    "streaks": "streaks",
    "dots_and_globules": "dots_globules",
    "regression_structures": "regression_structures",
}

# Explicit derm7pt taxonomy per concept (lowercased). We encode BINARY presence of
# the *suspicious* variant (the one that scores on the 7-point checklist). Listing
# the full known value set makes the mapping auditable and lets us warn only on
# genuinely unexpected values. Verified against the observed meta.csv categories.
_POSITIVE_VALUES: dict[str, set[str]] = {
    "pigment_network": {"atypical"},
    "blue_whitish_veil": {"present"},
    "vascular_structures": {"within regression", "dotted", "linear irregular"},
    "pigmentation": {"diffuse irregular", "localized irregular"},
    "streaks": {"irregular"},
    "dots_globules": {"irregular"},
    "regression_structures": {"blue areas", "white areas", "combinations"},
}
_NEGATIVE_VALUES: dict[str, set[str]] = {
    "pigment_network": {"absent", "typical"},
    "blue_whitish_veil": {"absent"},
    "vascular_structures": {"absent", "arborizing", "comma", "hairpin", "wreath",
                            "linear"},
    "pigmentation": {"absent", "diffuse regular", "localized regular"},
    "streaks": {"absent", "regular"},
    "dots_globules": {"absent", "regular"},
    "regression_structures": {"absent"},
}


def _binarize(concept_key: str, raw: str) -> tuple[float, bool]:
    """Map a raw derm7pt value to binary suspicious-present {0,1}.

    Returns (value, recognized) where recognized=False flags a value not in the
    known taxonomy (so the caller can warn). Unknown values default to 0.
    """
    v = str(raw).strip().lower()
    if v in ("", "nan"):
        return 0.0, True
    if v in _POSITIVE_VALUES.get(concept_key, set()):
        return 1.0, True
    if v in _NEGATIVE_VALUES.get(concept_key, set()):
        return 0.0, True
    return 0.0, False


def _group_diagnosis(raw: str) -> str:
    d = str(raw).strip().lower()
    if "basal cell carcinoma" in d:
        return "BCC"
    if "melanoma" in d:
        return "MEL"
    if "nevus" in d:
        return "NEV"
    if "seborrheic keratosis" in d:
        return "SK"
    return "MISC"


DX_ORDER = ["BCC", "MEL", "NEV", "SK", "MISC"]


def _read_index_file(f) -> list[int]:
    """Robustly read a derm7pt split-index file (delimiter varies; may be
    comma, tab, or whitespace). Returns the integer indices in the 'ind' column,
    falling back to the first column if unnamed."""
    idx_df = pd.read_csv(f, sep=None, engine="python")
    idx_df.columns = [str(c).strip().lower() for c in idx_df.columns]
    col = "ind" if "ind" in idx_df.columns else idx_df.columns[0]
    return (
        pd.to_numeric(idx_df[col], errors="coerce").dropna().astype(int).tolist()
    )


def _assign_splits(df: pd.DataFrame, paths: dict) -> pd.Series:
    """Official derm7pt splits: index files hold 0-based row indices of meta.csv."""
    split = pd.Series("", index=df.index, dtype=object)
    for name, key in (("train", "train_idx"), ("val", "valid_idx"), ("test", "test_idx")):
        f = paths.get(key)
        if f is None or not f.exists():
            continue
        idx = _read_index_file(f)
        hit = df.index.intersection(idx)
        split.loc[hit] = name
    return split


def load(verbose: bool = True) -> pd.DataFrame:
    paths = config.dataset_paths("derm7pt")
    if paths is None:
        raise FileNotFoundError("derm7pt not found under INPUT_ROOT")
    meta = pd.read_csv(paths["meta_csv"])
    images_dir = paths["images_dir"]

    out = S.empty_frame()
    # Build row-by-row to keep the binary mapping explicit & auditable.
    rows = []
    unmapped: dict[str, set] = {}
    for i, r in meta.iterrows():
        rec = {c: "" for c in S.BASE_COLUMNS}
        rec["uid"] = f"derm7pt_{int(r['case_num'])}"
        rec["source"] = "derm7pt"
        # Dermoscopic image is primary (concepts are dermoscopic); clinic is alt.
        derm_rel = str(r.get("derm", "")).strip()
        clin_rel = str(r.get("clinic", "")).strip()
        rec["image_path"] = str(images_dir / derm_rel) if derm_rel and derm_rel != "nan" else ""
        rec["image_path_alt"] = str(images_dir / clin_rel) if clin_rel and clin_rel != "nan" else ""
        rec["dx_name"] = _group_diagnosis(r["diagnosis"])
        rec["dx_label"] = DX_ORDER.index(rec["dx_name"])
        rec["fitzpatrick"] = np.nan
        rec["fst_group"] = ""
        rec["patient_id"] = f"derm7pt_{int(r['case_num'])}"   # one case ~ one patient
        rec["lesion_id"] = f"derm7pt_{int(r['case_num'])}"
        rec["split"] = ""
        # concepts
        for src_col, key in DERM7PT_COL.items():
            raw = r.get(src_col, "")
            val, recognized = _binarize(key, raw)
            rec[S.concept_col(key)] = val
            rec[S.mask_col(key)] = 1
            if not recognized:
                unmapped.setdefault(key, set()).add(str(raw))
        # concepts NOT in derm7pt (asymmetry) -> mask 0
        for key in C.CONCEPT_KEYS:
            if S.mask_col(key) not in rec:
                rec[S.concept_col(key)] = np.nan
                rec[S.mask_col(key)] = 0
        rows.append(rec)

    df = pd.DataFrame(rows)
    df = S.validate(df)
    df.index = meta.index                      # keep meta row index for split join
    df["split"] = _assign_splits(df, paths)

    if verbose:
        print(f"[derm7pt] {len(df)} cases")
        print(f"[derm7pt] dx distribution:\n{df['dx_name'].value_counts().to_string()}")
        print(f"[derm7pt] split counts: {df['split'].value_counts().to_dict()}")
        conc = df[[S.concept_col(k) for k in DERM7PT_COL.values()]].mean().round(3)
        print(f"[derm7pt] concept positive-rate:\n{conc.to_string()}")
        n_missing_img = (~df['image_path'].apply(lambda p: bool(p))).sum()
        print(f"[derm7pt] rows w/o derm image path: {n_missing_img}")
        if unmapped:
            warnings.warn(f"[derm7pt] unmapped concept values (check mapping): {unmapped}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    load()
