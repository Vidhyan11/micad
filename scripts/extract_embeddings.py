"""ME driver: extract & cache frozen-encoder embeddings for each dataset.

Run once per encoder (embeddings are reused by every downstream step):

    !python /kaggle/working/micad/scripts/extract_embeddings.py --encoder dinov2
    !python /kaggle/working/micad/scripts/extract_embeddings.py --encoder dermlip

Outputs (under artifacts/embeddings/):
    emb_<dataset>_<encoder>.npy   (N, dim) float32, L2-normalized
    meta_<dataset>.parquet        aligned rows (unified schema) + 'emb_valid' col
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.data import LOADERS  # noqa: E402
from fbc.encoders import build_encoder  # noqa: E402
from fbc.utils import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2", choices=["dinov2", "dermlip", "monet"])
    ap.add_argument("--datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per dataset")
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encoder={args.encoder}")
    enc = build_encoder(args.encoder, device=device)

    for ds in args.datasets:
        print(f"\n===== {ds} =====")
        df = LOADERS[ds](verbose=False)
        df = df[df["image_path"].astype(str) != ""].reset_index(drop=True)
        if args.limit:
            df = df.head(args.limit).reset_index(drop=True)
        print(f"[{ds}] embedding {len(df)} images...")
        emb, valid = enc.embed_paths(df["image_path"].tolist(), batch_size=args.batch_size)
        df["emb_valid"] = valid
        n_bad = int((~valid).sum())
        emb_file = io.emb_path(ds, args.encoder)
        meta_file = io.meta_path(ds)
        np.save(emb_file, emb)
        df.to_parquet(meta_file)
        print(f"[{ds}] saved {emb.shape} -> {emb_file.name} "
              f"(dim={enc.dim}, failed images={n_bad})")
        print(f"[{ds}] meta -> {meta_file.name}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
