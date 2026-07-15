"""Probe the 4 attached Kaggle datasets: print directory tree + CSV headers so we
can write correct loaders. Run this in a Kaggle cell (internet not required):

    !python /kaggle/working/micad/scripts/probe_datasets.py
    # or paste the body into a cell

Paste the full output back. It resolves PLAN.md §11 (exact columns/layout).
"""
from __future__ import annotations

import os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

# Logical name -> expected mount dirname (from fbc.config.DATASETS)
MOUNTS = {
    "derm7pt": "derm7pt",
    "ph2": "melanoma-skin-lesion-id-ph2-data",
    "fitzpatrick17k": "fitzpatrick17k",
    "pad_ufes_20": "pad-ufes-20",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tree(root: Path, max_depth: int = 3, max_entries: int = 40) -> None:
    """Compact directory tree with per-dir file counts and image counts."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        rel = Path(dirpath).relative_to(root)
        indent = "  " * depth
        n_img = sum(1 for f in filenames if Path(f).suffix.lower() in IMG_EXT)
        others = [f for f in filenames if Path(f).suffix.lower() not in IMG_EXT]
        label = f"{indent}{rel if str(rel) != '.' else root.name}/"
        print(f"{label}  [{len(filenames)} files, {n_img} images]")
        for f in sorted(others)[:max_entries]:
            print(f"{indent}  - {f}")
        if len(others) > max_entries:
            print(f"{indent}  ... (+{len(others) - max_entries} more non-image files)")


def dump_csvs(root: Path, max_rows: int = 3) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("  (pandas unavailable)")
        return
    tabular = list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + \
        list(root.rglob("*.txt"))
    for f in tabular:
        print(f"\n  >>> {f.relative_to(root)}")
        try:
            if f.suffix.lower() == ".xlsx":
                df = pd.read_excel(f, nrows=200)
            else:
                df = pd.read_csv(f, nrows=200, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            print(f"      (could not parse: {e})")
            continue
        print(f"      shape(first200)={df.shape}")
        print(f"      columns={list(df.columns)}")
        print(df.head(max_rows).to_string(max_colwidth=40))


def main() -> None:
    print("=" * 78)
    print("KAGGLE INPUT CONTENTS:", sorted(p.name for p in INPUT_ROOT.iterdir())
          if INPUT_ROOT.exists() else "(/kaggle/input missing)")
    print("=" * 78)
    for name, mount in MOUNTS.items():
        root = INPUT_ROOT / mount
        print(f"\n\n########## {name}  ->  {root}")
        if not root.exists():
            # try to find a close match
            cands = [p.name for p in INPUT_ROOT.iterdir()] if INPUT_ROOT.exists() else []
            print(f"  NOT FOUND. Available mounts: {cands}")
            continue
        tree(root)
        dump_csvs(root)


if __name__ == "__main__":
    main()
