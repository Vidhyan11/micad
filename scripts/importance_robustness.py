"""T2: is the faithfulness result specific to gradient attribution?

Recompute the pure bottleneck's COMPREHENSIVENESS (drop when the top stated-important
concepts are ablated) using several importance methods -- gradient, leave-one-out (loo),
and a random control. If gradient ~ loo and both >> random, the conclusion is not an
artifact of the gradient attribution, and the model's stated-important concepts are
genuinely the load-bearing ones.

    !python /kaggle/working/micad/scripts/importance_robustness.py --encoder dermlip --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.eval.bootstrap import fmt_ms, mean_std  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

SPECS = {
    "A": {"ds": "derm7pt", "use_pseudo": False, "binary_positive": "MEL"},
    "B": {"ds": "fitzpatrick17k", "use_pseudo": True, "binary_positive": "malignant"},
}
METHODS = ["gradient", "loo", "random"]


def run_seed(spec, args, cfg, device):
    data = assemble(spec["ds"], args.encoder, use_pseudo=spec["use_pseudo"],
                    binary_positive=spec["binary_positive"])
    Xtr = torch.as_tensor(data.emb[data.split_mask("train")]).to(device)
    Xte = torch.as_tensor(data.emb[data.split_mask("test")]).to(device)
    pure = T.train_cbm(data, cfg, device, mode="sequential")
    with torch.no_grad():
        ref = pure.predict_concepts(Xtr).mean(0)
        cp_te = pure.predict_concepts(Xte)
    fn = lambda c: pure.diagnose_from_concepts(c)
    return {m: faithfulness_scores(cp_te, fn, ref, importance=m)["comprehensiveness"]
            for m in METHODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    print(f"device={device} encoder={args.encoder} seeds={args.seeds}")

    rows = []
    for key in args.models:
        spec = SPECS[key]
        per = defaultdict(list)
        for seed in args.seeds:
            out = run_seed(spec, args, replace(base, seed=seed), device)
            for m in METHODS:
                per[m].append(out[m])
        row = {"model": key, "dataset": spec["ds"]}
        for m in METHODS:
            mu, sd = mean_std(per[m]); row[f"comp_{m}"] = mu; row[f"comp_{m}_std"] = sd
        rows.append(row)
        print(f"  Model {key}: comprehensiveness  "
              + "  ".join(f"{m}={fmt_ms(*mean_std(per[m]))}" for m in METHODS))

    tbl = pd.DataFrame(rows)
    out = io.result_path("exp_importance_robustness")
    tbl.to_csv(out, index=False)
    print(f"\n=== T2: comprehensiveness by importance method (mean+/-std over {len(args.seeds)} seeds) ===")
    print(tbl.to_string(index=False))
    print("\nExpect: gradient ~ loo (real methods agree) and both > random (importance is meaningful).")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
