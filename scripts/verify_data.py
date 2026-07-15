"""Run all loaders against the real Kaggle data and print a health report.
Confirms: schema validity, image-path resolution, concept/diagnosis/Fitzpatrick
distributions, and split assignment. Run in Kaggle after cloning the repo:

    !cd /kaggle/working/micad && git pull -q
    !python /kaggle/working/micad/scripts/verify_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# make `fbc` importable whether or not pip-installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fbc.config as C          # noqa: E402
from fbc.data import LOADERS, schema as S  # noqa: E402


def check_images(df, name, n=3):
    have = (df["image_path"].astype(str) != "") & df["image_path"].apply(
        lambda p: bool(p) and Path(p).exists())
    print(f"[{name}] image files that actually exist on disk: {have.sum()}/{len(df)}")
    missing = df.loc[~have, "image_path"].head(n).tolist()
    if (~have).any():
        print(f"[{name}] example unresolved image_path values: {missing}")


def main():
    print("ON_KAGGLE:", C.ON_KAGGLE, "| INPUT_ROOT:", C.INPUT_ROOT)
    print("available_datasets:", C.available_datasets())
    print("=" * 70)
    for name, load in LOADERS.items():
        print(f"\n########## {name}")
        try:
            df = load(verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {type(e).__name__}: {e}")
            continue
        # schema sanity
        assert list(df.columns) == S.ALL_COLUMNS, f"{name}: column mismatch"
        print(f"[{name}] rows={len(df)}, columns OK ({len(df.columns)})")
        check_images(df, name)


if __name__ == "__main__":
    main()
