"""Assemble training tensors for a model from cached embeddings + meta + pseudo.

Returns a dict of numpy arrays restricted to kept, embedding-valid rows, with the
domain concept targets/masks and split labels. Model A (derm7pt) uses GT concept
targets; Model B (Fitzpatrick/PAD) uses foundation pseudo-labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import concepts as CC, schema as S
from ..utils import io


@dataclass
class Assembled:
    emb: np.ndarray            # (N, d)
    concept_targets: np.ndarray  # (N, k) in [0,1]
    concept_mask: np.ndarray     # (N, k) 0/1
    y: np.ndarray              # (N,) dx label
    split: np.ndarray          # (N,) "train"/"val"/"test"/""
    keys: list                 # concept keys (domain order)
    n_classes: int

    def subset(self, split_name: str):
        m = self.split == split_name
        return (self.emb[m], self.concept_targets[m], self.concept_mask[m], self.y[m])


def assemble(ds: str, encoder: str, use_pseudo: bool,
             binary_positive: str | None = None) -> Assembled:
    """Assemble tensors. If binary_positive is a dx_name (e.g. 'MEL', 'malignant'),
    the task becomes binary detection (that class vs rest); else full multiclass."""
    df = pd.read_parquet(io.meta_path(ds))
    emb = np.load(io.emb_path(ds, encoder))
    assert len(df) == len(emb), f"{ds}: meta/emb length mismatch"

    domain = CC.DATASET_DOMAIN[ds]
    keys = CC.keys_for_domain(domain)
    kidx = [CC.index_of(k) for k in keys]

    if use_pseudo:
        pseudo = np.load(io.pseudo_path(ds, encoder))          # (N, 14)
        targets = pseudo[:, kidx].astype(np.float32)
        mask = np.ones_like(targets, dtype=np.float32)
    else:
        targets = df[[S.concept_col(k) for k in keys]].to_numpy(np.float32)
        mask = df[[S.mask_col(k) for k in keys]].to_numpy(np.float32)
        targets = np.nan_to_num(targets)

    if binary_positive is not None:
        y_all = (df["dx_name"].astype(str) == binary_positive).astype(np.int64).to_numpy()
        n_classes = 2
    else:
        y_all = df["dx_label"].to_numpy().astype(np.int64)
        n_classes = int(df["dx_label"].max()) + 1

    keep = df["is_dup_rep"].to_numpy(bool)
    if "emb_valid" in df.columns:
        keep &= df["emb_valid"].to_numpy(bool)

    return Assembled(
        emb=emb[keep].astype(np.float32),
        concept_targets=targets[keep],
        concept_mask=mask[keep],
        y=y_all[keep],
        split=df["split"].to_numpy().astype(str)[keep],
        keys=list(keys),
        n_classes=n_classes,
    )
