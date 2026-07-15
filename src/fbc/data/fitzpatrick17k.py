"""Fitzpatrick17k loader — FAIRNESS dataset with strong dark-skin coverage
(~16.5k images, Fitzpatrick I-VI). No dermoscopic concept GT -> concept masks 0
(pseudo-labels filled in MP). Diagnosis is the coarse `three_partition_label`
(benign / malignant / non-neoplastic); the fine 114-class `label` is not aligned
to the derm7pt diagnosis head, so we keep the coarse partition for reference.

Images are matched to CSV rows by md5hash (image filename stem == md5hash).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import concepts as C
from . import schema as S


def _fst_group(scale) -> str:
    try:
        f = int(round(float(scale)))
    except (TypeError, ValueError):
        return ""
    for name, members in config.FST_GROUPS.items():
        if f in members:
            return name
    return ""                      # -1 / unknown / out-of-range


def _build_image_index(images_root) -> dict[str, str]:
    """stem -> path for all images under images_root (handles subfolders)."""
    idx: dict[str, str] = {}
    if images_root is None or not images_root.exists():
        return idx
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in images_root.rglob(ext):
            idx.setdefault(p.stem, str(p))
    return idx


def load(verbose: bool = True) -> pd.DataFrame:
    paths = config.dataset_paths("fitzpatrick17k")
    if paths is None:
        raise FileNotFoundError("fitzpatrick17k not found under INPUT_ROOT")
    meta = pd.read_csv(paths["csv"])
    img_index = _build_image_index(paths["images_root"])

    # coarse diagnosis (stable order)
    part_col = "three_partition_label"
    dx_values = sorted(meta[part_col].astype(str).str.strip().unique()) \
        if part_col in meta.columns else ["unknown"]
    dx_order = {d: i for i, d in enumerate(dx_values)}

    rows = []
    n_img_missing = 0
    for i, r in meta.iterrows():
        h = str(r["md5hash"]).strip()
        path = img_index.get(h, "")
        if not path:
            n_img_missing += 1
        fitz = r.get("fitzpatrick_scale", np.nan)
        dx = str(r.get(part_col, "unknown")).strip()
        rec = {c: "" for c in S.BASE_COLUMNS}
        rec.update(
            uid=f"fitz_{i}",
            source="fitzpatrick17k",
            image_path=path,
            image_path_alt="",
            dx_name=dx,
            dx_label=dx_order.get(dx, 0),
            fitzpatrick=float(fitz) if pd.notna(fitz) else np.nan,
            fst_group=_fst_group(fitz),
            patient_id=h,
            lesion_id=h,
            split="",
        )
        for key in C.CONCEPT_KEYS:
            rec[S.concept_col(key)] = np.nan
            rec[S.mask_col(key)] = 0
        rows.append(rec)

    df = S.validate(pd.DataFrame(rows))
    # rows with a valid FST scale (1..6) and a resolved image are the usable set
    df["fitzpatrick"] = pd.to_numeric(df["fitzpatrick"], errors="coerce")

    if verbose:
        print(f"[fitzpatrick17k] {len(df)} rows; images resolved: "
              f"{len(df) - n_img_missing}/{len(df)}")
        print(f"[fitzpatrick17k] coarse dx:\n{df['dx_name'].value_counts().to_string()}")
        fst = df['fst_group'].replace("", "unknown").value_counts()
        print(f"[fitzpatrick17k] Fitzpatrick group counts:\n{fst.to_string()}")
        usable = ((df['fst_group'] != "") & (df['image_path'] != "")).sum()
        print(f"[fitzpatrick17k] usable (FST known + image found): {usable}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    load()
