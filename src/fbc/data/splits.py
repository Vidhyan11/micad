"""Leakage-clean protocol (contribution 4): embedding-based near-duplicate
detection + group-aware, stratified train/val/test splits.

- Duplicates are found by cosine similarity on cached (L2-normalized) embeddings
  and clustered with union-find; one representative per cluster is kept.
- Splits are GROUP-aware (no patient/lesion crosses a boundary) and stratified by
  diagnosis. Datasets with an official split (derm7pt) keep it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cosine_duplicate_clusters(
    emb: np.ndarray, valid: np.ndarray | None = None,
    threshold: float = 0.98, block: int = 2048,
) -> np.ndarray:
    """Return an array cluster_id (N,) grouping near-duplicate rows.

    Rows with valid=False are left as singletons. `emb` is assumed L2-normalized
    (re-normalized here defensively).
    """
    n = emb.shape[0]
    X = emb.astype(np.float32).copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-8, None)
    if valid is None:
        valid = norms.squeeze(1) > 1e-6
    uf = _UnionFind(n)
    for start in range(0, n, block):
        bx = X[start:start + block]                 # (b, d)
        sims = bx @ X.T                             # (b, n)
        for il in range(bx.shape[0]):
            i = start + il
            if not valid[i]:
                continue
            js = np.where(sims[il] >= threshold)[0]
            for j in js:
                if j > i and valid[j]:
                    uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def add_dedup_flags(
    df: pd.DataFrame, emb: np.ndarray, threshold: float = 0.98,
) -> pd.DataFrame:
    """Add 'dup_cluster' and 'is_dup_rep' (True = keep) columns."""
    valid = df["emb_valid"].to_numpy() if "emb_valid" in df.columns else None
    clusters = cosine_duplicate_clusters(emb, valid=valid, threshold=threshold)
    df = df.copy()
    df["dup_cluster"] = clusters
    # keep the first row of each cluster; drop later near-duplicates
    df["is_dup_rep"] = ~df.duplicated(subset="dup_cluster", keep="first")
    if valid is not None:
        df.loc[~df["emb_valid"], "is_dup_rep"] = False
    return df


# --------------------------------------------------------------------------- #
# Group-aware stratified split
# --------------------------------------------------------------------------- #
def stratified_group_split(
    df: pd.DataFrame, y_col: str = "dx_label", group_col: str = "patient_id",
    val_frac: float = 0.15, test_frac: float = 0.20, seed: int = 1337,
) -> pd.Series:
    """Assign train/val/test at the GROUP level, stratified by group-majority label.
    Returns a split Series aligned to df.index. Falls back to unstratified grouping
    if a class has too few groups to stratify.
    """
    from sklearn.model_selection import train_test_split

    maj = df.groupby(group_col)[y_col].agg(lambda s: s.value_counts().index[0])
    groups = maj.index.to_numpy()
    gy = maj.to_numpy()

    def _split(items, labels, frac):
        try:
            a, b = train_test_split(items, test_size=frac, stratify=labels,
                                    random_state=seed)
        except ValueError:                      # too few per class to stratify
            a, b = train_test_split(items, test_size=frac, random_state=seed)
        return a, b

    gy_map = dict(zip(groups, gy))
    trainval_g, test_g = _split(groups, gy, test_frac)
    tv_labels = np.array([gy_map[g] for g in trainval_g])
    rel_val = val_frac / (1.0 - test_frac)
    train_g, val_g = _split(trainval_g, tv_labels, rel_val)

    assign = {g: "train" for g in train_g}
    assign.update({g: "val" for g in val_g})
    assign.update({g: "test" for g in test_g})
    return df[group_col].map(assign).astype(object)


def assign_splits(
    df: pd.DataFrame, emb: np.ndarray | None = None,
    dedup_threshold: float = 0.98, seed: int = 1337,
) -> pd.DataFrame:
    """Full protocol: dedup flags (if emb given) + splits (keep official if present)."""
    if emb is not None:
        df = add_dedup_flags(df, emb, threshold=dedup_threshold)
    else:
        df = df.copy()
        df["dup_cluster"] = np.arange(len(df))
        df["is_dup_rep"] = True

    has_official = (df["split"].astype(str) != "").any()
    if not has_official:
        # split only over kept representatives; dropped dupes get "" (excluded)
        rep = df["is_dup_rep"]
        split = pd.Series("", index=df.index, dtype=object)
        split.loc[rep] = stratified_group_split(df.loc[rep], seed=seed)
        df["split"] = split
    return df
