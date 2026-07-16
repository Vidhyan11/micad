"""PAD-UFES-20 loader — the FAIRNESS dataset (Fitzpatrick I-VI + smartphone
clinical images + rich metadata). No dermoscopic concept GT, so all concept masks
are 0 here; concept values are filled later by the foundation pseudo-labeler (MP).

Diagnosis: 6 classes from the `diagnostic` column. Skin tone: `fitspatrick` (note
the dataset's spelling) 1..6, grouped into I-II / III-IV / V-VI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import concepts as C
from . import schema as S


def _fst_group(fitz: float) -> str:
    if fitz is None or (isinstance(fitz, float) and np.isnan(fitz)):
        return ""
    f = int(round(fitz))
    for name, members in config.FST_GROUPS.items():
        if f in members:
            return name
    return ""


def _as_bool01(v) -> float | None:
    """Coerce PAD-UFES boolean-ish metadata to {0.0, 1.0} or None if missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "1.0"):
        return 1.0
    if s in ("false", "0", "no", "0.0"):
        return 0.0
    return None


def _build_image_index(images_root) -> dict[str, str]:
    """Map image filename -> absolute path (scan imgs_part_* once)."""
    idx: dict[str, str] = {}
    if images_root is None or not images_root.exists():
        return idx
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in images_root.rglob(ext):
            idx[p.name] = str(p)
    return idx


def load(verbose: bool = True) -> pd.DataFrame:
    paths = config.dataset_paths("pad_ufes_20")
    if paths is None:
        raise FileNotFoundError("pad_ufes_20 not found under INPUT_ROOT")
    meta = pd.read_csv(paths["metadata_csv"])
    img_index = _build_image_index(paths["images_root"])

    # stable diagnosis ordering
    dx_values = sorted(meta["diagnostic"].astype(str).str.strip().unique())
    dx_order = {d: i for i, d in enumerate(dx_values)}

    rows = []
    n_img_missing = 0
    for i, r in meta.iterrows():
        rec = {c: "" for c in S.BASE_COLUMNS}
        img_id = str(r["img_id"]).strip()
        path = img_index.get(img_id, "")
        if not path:
            n_img_missing += 1
        dx = str(r["diagnostic"]).strip()
        fitz = r.get("fitspatrick", np.nan)
        try:
            fitz = float(fitz)
        except (TypeError, ValueError):
            fitz = np.nan
        rec.update(
            uid=f"pad_{i}",
            source="pad_ufes_20",
            image_path=path,
            image_path_alt="",
            dx_name=dx,
            dx_label=dx_order[dx],
            fitzpatrick=fitz,
            fst_group=_fst_group(fitz),
            patient_id=str(r.get("patient_id", f"pad_pat_{i}")),
            lesion_id=str(r.get("lesion_id", f"pad_les_{i}")),
            split="",
        )
        # no dermoscopic concept GT; clinical concepts mostly pseudo-labeled later.
        for key in C.CONCEPT_KEYS:
            rec[S.concept_col(key)] = np.nan
            rec[S.mask_col(key)] = 0
        # PARTIAL clinical GT from metadata -> used to VALIDATE pseudo-labels (MP).
        elev = _as_bool01(r.get("elevation"))
        if elev is not None:
            rec[S.concept_col("elevation")] = elev
            rec[S.mask_col("elevation")] = 1
        rows.append(rec)

    df = S.validate(pd.DataFrame(rows))

    if verbose:
        print(f"[pad_ufes_20] {len(df)} lesions; images resolved: {len(df) - n_img_missing}/{len(df)}")
        print(f"[pad_ufes_20] dx distribution:\n{df['dx_name'].value_counts().to_string()}")
        fst = df['fst_group'].replace("", "NaN").value_counts()
        print(f"[pad_ufes_20] Fitzpatrick group counts:\n{fst.to_string()}")
        print(f"[pad_ufes_20] fitspatrick NaN rate: "
              f"{df['fitzpatrick'].isna().mean():.2%}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    load()
