"""MD part 3 driver: apply the leakage-clean protocol to each cached dataset.

Reads meta_<ds>.parquet + emb_<ds>_<encoder>.npy (from ME), adds dedup flags and
train/val/test splits, writes them back into meta_<ds>.parquet, and prints a
splits_report we cite in the paper's protocol section.

    !python /kaggle/working/micad/scripts/make_splits.py --encoder dinov2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.data import LOADERS, splits as SP  # noqa: E402
from fbc.utils import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2")
    ap.add_argument("--datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    report = {}
    for ds in args.datasets:
        meta_f = io.meta_path(ds)
        emb_f = io.emb_path(ds, args.encoder)
        if not meta_f.exists() or not emb_f.exists():
            print(f"[{ds}] SKIP — run extract_embeddings first ({meta_f.name}, {emb_f.name})")
            continue
        df = pd.read_parquet(meta_f)
        emb = np.load(emb_f)
        assert len(df) == len(emb), f"{ds}: meta/emb length mismatch"

        df = SP.assign_splits(df, emb=emb, dedup_threshold=args.threshold, seed=args.seed)
        df.to_parquet(meta_f)

        n = len(df)
        n_rep = int(df["is_dup_rep"].sum())
        n_dup = n - n_rep
        split_counts = df.loc[df["is_dup_rep"], "split"].value_counts().to_dict()
        report[ds] = {"rows": n, "kept": n_rep, "near_dupes_dropped": n_dup,
                      "splits": split_counts}
        print(f"\n[{ds}] rows={n} kept={n_rep} near-dupes-dropped={n_dup}")
        print(f"[{ds}] split counts (reps): {split_counts}")
        # fairness sanity: FST group distribution per split (clinical datasets)
        if (df["fst_group"].astype(str) != "").any():
            xt = pd.crosstab(df.loc[df.is_dup_rep, "split"],
                             df.loc[df.is_dup_rep, "fst_group"])
            print(f"[{ds}] split x Fitzpatrick group:\n{xt.to_string()}")

    io.save_json(report, C.RESULT_DIR / "splits_report.json")
    print(f"\nsplits_report -> {C.RESULT_DIR / 'splits_report.json'}")


if __name__ == "__main__":
    main()
