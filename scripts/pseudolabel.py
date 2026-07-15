"""MP driver: foundation zero-shot concept pseudo-labels + validation.

Requires a CLIP-style encoder (DermLIP/MONET) with a text tower.

    !pip install -q open_clip_torch
    !python /kaggle/working/micad/scripts/pseudolabel.py --encoder dermlip

For each dataset, scores every canonical concept (pos/neg prompt pair) from the
cached image embeddings and saves pseudo_<ds>_<enc>.npy  (N, 15) probabilities.

Validation (the contribution-2 evidence): on derm7pt we HAVE real dermoscopic
concept GT, so we report zero-shot AUROC/AP per concept — showing the foundation
model recovers the concepts well enough to bootstrap supervision.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.data import LOADERS, concepts as CC, schema as S  # noqa: E402
from fbc.encoders import build_encoder  # noqa: E402
from fbc.utils import io  # noqa: E402


def pseudo_path(ds: str, enc: str) -> Path:
    return C.EMB_DIR / f"pseudo_{ds}_{enc}.npy"


def validate_against_gt(df: pd.DataFrame, probs: np.ndarray) -> dict:
    """Per-concept zero-shot AUROC/AP where GT exists (mask==1)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {}
    for ci, key in enumerate(CC.CONCEPT_KEYS):
        m = df[S.mask_col(key)].to_numpy() == 1
        if m.sum() < 10:
            continue
        y = df.loc[m, S.concept_col(key)].to_numpy().astype(int)
        p = probs[m, ci]
        if len(np.unique(y)) < 2:
            continue
        out[key] = {
            "n": int(m.sum()),
            "pos_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip", choices=["dermlip", "monet"])
    ap.add_argument("--datasets", nargs="*", default=list(LOADERS))
    ap.add_argument("--temperature", type=float, default=0.01)
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_encoder(args.encoder, device=device)
    enc.ensure_loaded()
    if not hasattr(enc, "zeroshot_concepts"):
        raise SystemExit(f"{args.encoder} has no text tower; use dermlip or monet")

    report = {}
    for ds in args.datasets:
        emb_f = io.emb_path(ds, args.encoder)
        meta_f = io.meta_path(ds)
        if not emb_f.exists():
            print(f"[{ds}] SKIP — extract {args.encoder} embeddings first ({emb_f.name})")
            continue
        emb = np.load(emb_f)
        df = pd.read_parquet(meta_f)
        probs = enc.zeroshot_concepts(emb, temperature=args.temperature)  # (N, 15)
        np.save(pseudo_path(ds, args.encoder), probs)
        print(f"[{ds}] pseudo-labels {probs.shape} -> {pseudo_path(ds, args.encoder).name}")

        domain = CC.DATASET_DOMAIN.get(ds)
        # validate where we have GT (derm7pt dermoscopic concepts)
        val = validate_against_gt(df, probs)
        if val:
            print(f"[{ds}] zero-shot concept recovery vs GT ({args.encoder}):")
            print(f"  {'concept':24s} {'n':>5s} {'pos%':>6s} {'AUROC':>6s} {'AP':>6s}")
            for k, v in val.items():
                print(f"  {k:24s} {v['n']:5d} {v['pos_rate']*100:5.1f} "
                      f"{v['auroc']:6.3f} {v['ap']:6.3f}")
            mean_auroc = float(np.mean([v["auroc"] for v in val.values()]))
            print(f"  mean AUROC = {mean_auroc:.3f}")
            report[ds] = {"mean_auroc": mean_auroc, "per_concept": val}
        else:
            # clinical datasets: report pseudo positive-rate for its vocabulary
            clin = {k: float((probs[:, CC.index_of(k)] > 0.5).mean())
                    for k in CC.CLINICAL_KEYS}
            print(f"[{ds}] clinical pseudo-concept positive-rate (domain={domain}):")
            for k, v in clin.items():
                print(f"  {k:24s} {v*100:5.1f}%")
            report[ds] = {"clinical_pos_rate": clin}

    io.save_json(report, C.RESULT_DIR / f"pseudolabel_report_{args.encoder}.json")
    print(f"\nreport -> {C.RESULT_DIR / f'pseudolabel_report_{args.encoder}.json'}")


if __name__ == "__main__":
    main()
